const { RecursiveCharacterTextSplitter } = require('@langchain/textsplitters');
const {ChatGoogleGenerativeAI  }=require('@langchain/google-genai');
const {CohereEmbeddings} =require("@langchain/cohere")
const { MongoClient } = require('mongodb');
const Bottleneck = require('bottleneck');

const limiter = new Bottleneck({
  reservoir:           10,             // initial number of available calls
  reservoirRefreshAmount: 10,          // number of calls to restore
  reservoirRefreshInterval: 30 * 1000, // interval in ms to restore
});

const embeddings = new CohereEmbeddings({
  apiKey: process.env.COHERE_API_KEY,
  batchSize: 48,
  model: "embed-english-v3.0",
});

const client = new MongoClient(process.env.ATLAS_CONNECTION_STRING, {
  maxPoolSize: 50,
  w: 'majority',
});

const llm = new ChatGoogleGenerativeAI({
  model:       "gemini-2.5-flash",
  apiKey:      process.env.GOOGLE_API_KEY,
  temperature: 0.1,
});

async function RecursiveSplitting(context,collection){
    await vectorIndexCreate(collection)
    let charsplit=new RecursiveCharacterTextSplitter({chunkSize:1024,chunkOverlap:100})
    const text=await charsplit.createDocuments([context])
    console.log(text.length)
    try {
        console.log("Sending data to Graph API...");
        const graphApiUrl = process.env.GRAPH_API_URL || 'http://127.0.0.1:8000';
        const response = await axios.post(`${graphApiUrl}/insert`, {
            text: text
        });
        console.log("Response from Graph API:", response.data);
    } catch (error) {
      console.error("Error connecting to Graph API:", error.message);
    }
    const {embedding,boundries}=await limiter.schedule(()=> Embedding(text)) 
    const data=merge(embedding,text,boundries)
    await SaveInDataBaseMongo(data,collection)
    console.log("saved successfully")
    return 
}

async function Embedding(context){
    console.log("Embedding starts.....")
    const texts = context.map(doc => doc.pageContent);
    const documentRes = await embeddings.embedDocuments(texts);
    return FindSimilarChunk(documentRes,context)
}

async function FindSimilarChunk(embedding,context){
    const boundries=[0]

    for(let i = 1; i < embedding.length; i++){
        // Compare current embedding with previous one
        const sim = dotProduct(embedding[i-1], embedding[i])
        if(sim > 0.6) boundries.push(i)
    }
    boundries.push(embedding.length)
    console.log('Boundries created!')

    return {embedding,boundries}
}

function dotProduct(arr1, arr2) {
    // Check if both inputs are arrays
    if (!Array.isArray(arr1) || !Array.isArray(arr2)) {
      return "Error: Both parameters must be arrays.";
    }
  
    // Check if the arrays have the same length
    if (arr1.length !== arr2.length) {
      return "Error: Arrays must be of the same length to calculate the dot product.";
    }
  
    let product = 0;
    // Iterate through the arrays and sum the products of corresponding elements
    for (let i = 0; i < arr1.length; i++) {
      // Ensure elements are numbers for calculation
      if (typeof arr1[i] !== 'number' || typeof arr2[i] !== 'number') {
        return "Error: All elements in arrays must be numbers.";
      }
      product += arr1[i] * arr2[i];
    }
  
    return product;
}

function merge(embedding,text,boundries){
    const docs=[]
    const merged = Object.keys(embedding).map(key => {
        // ensure the chunk text exists
        if (!(key in text)) {
          throw new Error(`No text found for chunk ${key}`);
        }
        return {
          chunk_index: Number(key),
          text:        text[key],
          embedding:   embedding[key],
          // sparse_embeddings:sparse_embeddings[key]
        };
    });

    for(let i=0;i<boundries.length-1;i++){
        let startingind=boundries[i]
        let endind=boundries[i+1]-1
        for(let j=startingind;j<=endind;j++){
            const base = JSON.parse(JSON.stringify(merged[j]));
            const doc = {
              ...base,
              segment_id:    i,
              segment_start: startingind,
              segment_end:   endind
            };
            docs.push(doc)
        }
    }
    return docs
}

async function SaveInDataBaseMongo(data,collection){
   try {
     const options = { ordered: false,upsert:true };
     const result = await collection.insertMany(data, options);
     console.log("Count of documents inserted: " + result.insertedCount); 
   } catch (error) {
    console.log("Error while saving in DB",error)
   }
   finally {
    // vectorIndexCreate(collection)
    return
  }
}

async function vectorIndexCreate(collection){
  const vectorindex = {
    name: "dense_embedding",
    type: "vectorSearch",
    definition: {
      "fields": [
        {
          "type": "vector",
          "numDimensions": 1024,
          "path": "embedding",
          "similarity": "cosine"
        },
        // {
        //   "type":"filter",
        //   "path":""
        // }
      ]
    }
  }

  const similarityindex = {
    name: "sparse_embedding",
    type: "search",
    definition: {
      "mappings": {
        "dynamic": false,
        "fields": {
          "text": {
            "type": "document",
            "fields": {
              "pageContent": {
                "type": "string",
                "analyzer": "lucene.english",
                "searchAnalyzer": "lucene.english"
              }
            }
          },
          "segment_id": {
            "type": "number"
          },
          "segment_start": {
            "type": "number"
          },
          "segment_end": {
            "type": "number"
          }
        }
      }
    }
  }

  try {
    const result1 = await collection.createSearchIndex(vectorindex);
    const result2 = await collection.createSearchIndex(similarityindex);
    console.log("Vector index created", result1);
    console.log("search index created",result2)
  } catch (error) {
    console.error("Error creating indexes:", error);
  }
  return;
}

// Seaching
async function QueryEmbedding(query){
 try {
   const densequeryembedding = await embeddings.embedQuery(query);
   const dembedding = await DenseRetrieveQuery(densequeryembedding,query)
  //  const sembedding = await SparseRetrieveQuery(query)
  //  const result = await mergeRetrieval(dembedding,sembedding)
   return dembedding;
 } catch (error) {
  console.log("error in querying",error)
 }
}

async function DenseRetrieveQuery(queryembedding,userquery){
  let results
    try {
      await client.connect()
      const db=client.db("learn_vector")
      const collection=db.collection("learn_vector")
      const pipeline= [
          {
              $vectorSearch: {
                  index: "dense_embedding",
                  queryVector: queryembedding,
                  path: "embedding",
                  numCandidates: 100,
                  limit: 12,
                  similarityMetric: "cosine"  
              },
            },
            {
              $project: {
                  _id: 0,
                  text: "$text",
                  chunk_index:"$chunk_index",
                  score: { $meta: "vectorSearchScore" }
              }
            }
      ];
    const cursor = collection.aggregate(pipeline);
    results = await cursor.toArray();

    console.log("Found documents:");
    return results;
  } catch (error) {
    console.log("Error in finding query",error)
  }finally{
    await client.close()
  }
}

async function SparseRetrieveQuery(userquery){
  let results
    try {
      await client.connect()
      const db=client.db("learn_vector")
      const collection=db.collection("learn_vector")
      const pipeline= [
          {
              $search: {
                  index: "sparse_embedding",
                  text: {
                    query: userquery,
                    path: "text.pageContent"
                  }
              }
          },
          {
              $limit: 8
          },
          {
              $project: {
                  _id: 0,
                  text: "$text",
                  chunk_index:"$chunk_index",
                  score: { $meta: "searchScore" }
              }
          },
      ];
    const cursor = collection.aggregate(pipeline);
    results = await cursor.toArray();

    console.log("Found documents:");
    return results;
  } catch (error) {
    console.log("Error in finding sparse query",error)
  }finally{
    await client.close()
  }
}

async function mergeRetrieval(dembedding){
  const chunkids = new Set();
  const result=[]
  dembedding.forEach((doc,i)=>{
    let score;
    const chunkid=doc.chunk_index
    let sparse_embedding=sembedding.findIndex((id)=>  (id.chunk_index==chunkid))
    if(sparse_embedding!=-1 || !sparse_embedding){
      score=1/(i+1) + 1/(sparse_embedding+1)
      result.push({
        denseembedding:doc.score,
        score:score,
        chunkid,
        densetext:doc.text.pageContent,
      })
    }else{
      score=1/(i+1);
      result.push({
        denseembedding:doc.score,
        score:score,
        chunkid,
        densetext:doc.text.pageContent,
      })
    }
    chunkids.add(chunkid)
  })
  // sembedding.forEach((doc,i)=>{
  //   if(!chunkids.has(doc.chunk_index)){
  //     score=1/(i+1)
  //     result.push({
  //       denseembedding:"null",
  //       sparseembedding:doc.score,
  //       score:score,
  //       chunkid:doc.chunk_index,
  //       densetext:"null",
  //       sparsetext:doc.text?.pageContent || "null"
  //     })
  //   }
  // })
  result
  .sort((a, b) => b.score - a.score)
  .slice(0, 8);
  console.log(result)
  return result
}

async function answerwithollama(userquery,ragresult,graphresult){
  console.log(userquery)
  const context = ragresult.map((chunk, i) => {
    return `[Chunk ${chunk.chunkid}]\n${chunk.text.pageContent || chunk.sparsetext}`;
  }).join('\n\n');

  const chatprompt = `
  You are a friendly, conversational AI assistant. Use only the context below to answer the user's question in a natural chat style—no formal report sections, just a clear, helpful reply.

  HYBRID RAG CONTEXT:
  ${context}

  KNOWLEDGE GRAPH CONTEXT:
  ${graphresult}

  USER QUESTION:
  "${userquery}"

  INSTRUCTIONS:
  - Keep it concise and on-point (2–4 sentences).
  - Speak as if you're chatting one-on-one: use "you" and "I" naturally.
  - Combine insights from both contexts if relevant, but don't mention "Hybrid RAG" or "Knowledge Graph" to the user.
  - If the context doesn't cover the answer, say "I'm not sure based on what I have here."
  `;
  try {
      console.log("LLM Model is starting...");
      const response = await llm.invoke(chatprompt);
      console.log("\n✅ Response from LLM Model:");
      console.log(response.response_metadata)
      return response.content;
  } catch (error) {
      console.error("Error getting response from LLM Model:", error);
      return "Sorry, I encountered an error while trying to answer your question.";
  }
}

module.exports={
    RecursiveSplitting,
    QueryEmbedding,
    answerwithollama
}