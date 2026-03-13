from fastapi import FastAPI
from pydantic import BaseModel
import sys
import os
import json
import uvicorn
from langchain_neo4j import Neo4jGraph
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.documents import Document
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_neo4j.chains.graph_qa.cypher import GraphCypherQAChain
from neo4j import GraphDatabase
import igraph as ig
import leidenalg
import numpy as np
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

app = FastAPI()

class QueryRequest(BaseModel):
    query: str

class InsertRequest(BaseModel):
    text: str


load_dotenv()

NEO4J_URI=os.environ.get("NEO4J_URI")
NEO4J_USERNAME=os.environ.get("NEO4J_USERNAME")
NEO4J_PASSWORD=os.environ.get("NEO4J_PASSWORD")
NEO4J_DATABASE=os.environ.get("NEO4J_DATABASE")

URI = NEO4J_URI
AUTH = (NEO4J_USERNAME, NEO4J_PASSWORD)
graph=Neo4jGraph(url=NEO4J_URI, username=NEO4J_USERNAME, password=NEO4J_PASSWORD,database=NEO4J_DATABASE)

llm = ChatGoogleGenerativeAI(
    api_key=os.environ.get("GEMINI_API_KEY"),
    model="gemini-2.5-flash",
    temperature=0
)

chain=GraphCypherQAChain.from_llm(
    llm=llm,
    graph=graph,
    verbose=True,
    return_intermediate_steps=True,
    allow_dangerous_requests=True
    )

def semantic_node_search(query, top_k=5):

    query_embedding = embedding_model.encode(str(query))

    nodes = graph.query("""
    MATCH (n)
    WHERE n.embedding IS NOT NULL
    RETURN n.id, n.embedding
    """)

    scored_nodes = []

    for node in nodes:

        node_id = str(node["n.id"])

        node_emb = np.array(node["n.embedding"])

        score = np.dot(query_embedding, node_emb) / (
            np.linalg.norm(query_embedding) * np.linalg.norm(node_emb)
        )

        scored_nodes.append((node_id, score))

    scored_nodes.sort(key=lambda x: x[1], reverse=True)

    return [n[0] for n in scored_nodes[:top_k]]

def extract_entities(query):

    prompt = f"""
    Extract the important entities from this question.

    Return a comma separated list.

    Question:
    {query}
    """

    response = llm.invoke(prompt)

    entities = response.content.split(",")

    return [e.strip() for e in entities]

def find_nodes(query):

    entities = extract_entities(query)

    nodes = set()

    for e in entities:
        results = semantic_node_search(e)
        nodes.update(results)

    return list(nodes)

def get_communities(nodes):

    result = graph.query("""
    MATCH (n)
    WHERE n.id IN $nodes
    MATCH (n)<-[:HAS_MEMBER]-(c:Community)
    RETURN DISTINCT c.summary
    """, {"nodes": nodes})

    return [r["c.summary"] for r in result]

def expand_graph_paths(nodes):

    result = graph.query("""
    MATCH path = (n)-[*1..3]-(m)
    WHERE n.id IN $nodes
    AND m.id IS NOT NULL
    AND NOT toString(m.id) =~ '^[0-9]+$'
    RETURN path
    LIMIT 15
    """, {"nodes": nodes})

    return result

def paths_to_text(paths):

    chains = []

    for p in paths:

        nodes = p["path"]

        chain = " -> ".join(str(n) for n in nodes)

        chains.append(chain)

    return chains

def rank_paths(query, paths, top_k=9):

    query_emb = embedding_model.encode(query)

    scored = []

    for p in paths:

        emb = embedding_model.encode(p)

        score = np.dot(query_emb, emb) / (
            np.linalg.norm(query_emb) * np.linalg.norm(emb)
        )

        scored.append((p, score))

    scored.sort(key=lambda x: x[1], reverse=True)

    return [p[0] for p in scored[:top_k]]

def build_context(query):

    nodes = find_nodes(query)
    print(nodes)

    paths = expand_graph_paths(nodes)
    print(paths)

    path_strings = paths_to_text(paths)
    print(path_strings)

    ranked_paths = rank_paths(query, path_strings)
    print(ranked_paths)

    communities = get_communities(nodes)
    print(communities)

    return {
        "paths": ranked_paths,
        "communities": communities
    }

def summarize_community(nodes):

    node_text = "\n".join(nodes)

    prompt = f"""
    You are analyzing a knowledge graph.

    These entities belong to the same community:

    {node_text}

    Explain the main concept connecting these entities in 2–3 sentences.
    """

    response = llm.invoke(prompt)

    return response.content

def graph_rag_answer(query):

    context = build_context(query)

    path_text = "\n".join(context["paths"])

    community_text = "\n".join(context["communities"])

    # prompt = f"""
    # Answer the question using the context below.

    # Context:
    # {community_text}
    # {path_text}

    # Question:
    # {query}

    # Write a concise answer of about 2–4 sentences depending on the
    # complexity of the question. Do not mention the context sources.
    # """

    # response = llm.invoke(prompt)

    return community_text

def InsertData(text):
    documents=[Document(page_content=text)]

    llm_transformer=LLMGraphTransformer(
        llm=llm,
        strict_mode=False
    )
    
    graph_documents=llm_transformer.convert_to_graph_documents(documents)
    print(graph_documents)
    graph.add_graph_documents(graph_documents)

    query = """
    MATCH (n)-[r]->(m)
    RETURN n.id AS source, m.id AS target
    """

    edges = graph.query(query)
    edge_list = [(e["source"], e["target"]) for e in edges]

    g = ig.Graph.TupleList(edge_list, directed=False)

    partition = leidenalg.find_partition(g,leidenalg.ModularityVertexPartition)

    community_ids = partition.membership
    nodes = g.vs["name"]

    community_map = dict(zip(nodes, community_ids))

    communities = graph.query("""
    MATCH (n)
    WHERE n.communityId IS NOT NULL
    RETURN n.communityId AS community, collect(n.id) AS nodes
    """)

    for node, cid in community_map.items():
        graph.query(
            """
            MATCH (n {id:$node})
            SET n.communityId = $cid
            """,
            {"node": node, "cid": cid}
        )

    community_summaries = []

    for c in communities:

        cid = c["community"]
        nodes = c["nodes"]

        summary = summarize_community(nodes)

        community_summaries.append({
            "community": cid,
            "summary": summary,
            "nodes": nodes
        })

    print(f"Community {cid}")
    print(summary)
    print("-----")

    for c in community_summaries:

        graph.query("""
        MERGE (c:Community {id:$cid})
        SET c.summary = $summary
        """, {
            "cid": c["community"],
            "summary": c["summary"]
        })

    for c in community_summaries:

        a=graph.query("""
        MATCH (n {communityId:$cid})
        MATCH (c:Community {id:$cid})
        MERGE (c)-[:HAS_MEMBER]->(n)
        """, {"cid": c["community"]})
    return "success"

def GetAnswer(query):

    response=graph_rag_answer(query)

    print(response)
    return response

@app.get("/")
def home():
    return {"message": "API working"}

@app.post("/insert")
async def insert_data_endpoint(request: InsertRequest):
    print("Received text:", len(request.text))
    result = InsertData(request.text)
    return {"status": "success", "result": result}

@app.post("/get")
async def get_answer_endpoint(request: QueryRequest):
    print(request.query)
    result = GetAnswer(request.query)
    return {"status": "success", "result": result}
