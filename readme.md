# 🌾 Sahaay AI – Multimodal RAG-Powered AI Farming Assistant

A multilingual, multimodal AI Agent for farmers that combines Hybrid RAG, GraphRAG, voice interaction, real-time market prices, and live web intelligence to deliver accurate, personalized agricultural guidance.

## 📌 Overview

Sahaay AI is a production-oriented AI system enabling farmers to interact through:

* **📞 Voice Calls**
* **💬 WhatsApp Chat**
* **🖼️ Image-Based Queries** (Pesticides / Crop Issues)

The platform integrates:

* ✔ Domain-specific agricultural knowledge (Hybrid RAG + GraphRAG)
* ✔ Knowledge Graph with community-based reasoning
* ✔ Live subsidies & scheme discovery
* ✔ Real-time crop market prices
* ✔ AI-initiated voice calls
* ✔ Accessibility & disability support
* ✔ Persistent personalization & memory

---

## ✨ Key Capabilities

### 🤖 Intelligent AI Agent
* Conversational, task-aware AI built for real farming scenarios.
* Contextual responses using memory & retrieved knowledge.
* Seamless switch between text, voice, and images.
* Tool-calling via OpenAI function calling — agent decides when to query the knowledge graph.

### 📚 Hybrid RAG + GraphRAG Knowledge Pipeline
* **Hybrid RAG** — combines dense vector search with BM25 keyword retrieval for accurate document-level answers.
* **GraphRAG** — builds a Neo4j knowledge graph from agricultural data, detects communities via Leiden algorithm, and retrieves structured relational context.
* Both contexts are merged and passed to the LLM in a single call for richer, grounded answers.
* Reduced hallucinations through dual-source knowledge grounding.

### 🧰 Dynamic Tool-Calling System
The AI Agent invokes tools based on user intent:

| Use Case | Tool |
| :--- | :--- |
| Latest subsidies / schemes | 🌐 Tavily Web Search & Scraping |
| Crop / commodity prices | 📈 MarketPrice Tool |
| Call request / outbound AI call | 📞 Twilio Call Tool |
| Farming knowledge queries | 📚 Hybrid RAG Retrieval |
| Graph-based knowledge queries | 🕸️ GraphRAG Tool (Neo4j) |

### 🕸️ GraphRAG — Knowledge Graph Intelligence
* Agricultural data is transformed into a Neo4j knowledge graph using LLM-based entity & relationship extraction.
* Leiden community detection groups related agricultural concepts (crops, states, seasons, conditions).
* Each community is summarized by an LLM for fast, context-rich retrieval.
* Semantic node search using sentence embeddings finds the most relevant graph nodes per query.
* Graph path expansion (1–3 hops) uncovers hidden relationships between entities.
* Exposed as a FastAPI microservice, called in parallel with Hybrid RAG.

### 🖼️ Multimodal Image Understanding
* Farmers upload pesticide or crop images.
* AI performs visual reasoning.
* Generates practical treatment guidance.

### 📞 AI-Initiated Voice Calls
* Farmers request calls directly from WhatsApp.
* AI triggers automated call flow.
* Fully conversational voice experience.

### 🧾 Post-Call Intelligence & Summaries
After each call:
* ✔ Conversation summary generated.
* ✔ Delivered via WhatsApp + SMS.
* ✔ Useful farmer information extracted.
* ✔ Stored for personalization.

### 🌍 Multilingual & Inclusive Design
* Supports multiple languages including Hindi.
* Accessibility-focused interaction patterns.
* Designed for diverse literacy & ability levels.

---

## 🏗️ System Architecture
```text
Farmer (Voice / WhatsApp / Images)
                |
                v
           AI Agent Core (Node.js)
           OpenAI GPT-4o + Tool Calling
                |
     ┌──────────┼──────────┬──────────┐
     v          v          v          v
Hybrid RAG   GraphRAG   Tavily   MarketPrice
(Vector+BM25) (Neo4j)  (Web Data) (Live Prices)
     |          |
     └────┬─────┘
          v
   Combined Context
          |
          v
   LLM Response Generation
          |
          v
  WhatsApp / Voice / SMS
```

---

## 🔄 RAG Pipeline
```text
Farmer Query
      ↓
Intent + Entity Detection
      ↓
┌─────────────────────────────┐
│  Hybrid RAG                 │  ← Vector + BM25 search
│  GraphRAG (FastAPI/Neo4j)   │  ← Community summaries + graph paths
└─────────────────────────────┘
      ↓  (parallel fetch)
Context Merging
      ↓
LLM Response Generation (single call)
      ↓
Farmer Response
```

* ✔ Domain-aware reasoning
* ✔ Relational knowledge via graph traversal
* ✔ Community-level agricultural insights
* ✔ Knowledge-grounded outputs

---

## 🔄 Interaction Modes

### 💬 WhatsApp Interaction
* Ask farming & crop questions.
* Request latest schemes & subsidies.
* Upload images for issue diagnosis.
* Trigger AI voice calls.

### 📞 Voice Call Interaction
* Natural conversational AI.
* Farmer-friendly dialogue design.
* Low-friction usability.

### 🖼️ Image-Based Queries
* Upload pesticide / crop images.
* AI analyzes symptoms visually.
* Returns actionable advice.

---

## 🧠 Personalization Engine

The system continuously improves using interaction data:
* ✔ Extracts farmer-specific insights.
* ✔ Stores conversational context.
* ✔ Delivers smarter future responses.

**Benefits:**
* Personalized recommendations.
* Context-aware assistance.
* Improved accuracy over time.

---

## 🛠️ Tech Stack

### 🧠 AI & Intelligence
* GPT-4o — LLM with OpenAI Tool Calling
* Gemini 2.5 Flash — GraphRAG LLM + Entity Extraction
* Hybrid RAG — Vector Search + BM25
* GraphRAG — Neo4j + Leiden Community Detection
* Sentence Transformers — Semantic Node Embeddings

### 🕸️ Knowledge Graph
* Neo4j — Graph Database
* LangChain LLMGraphTransformer — Entity & Relation Extraction
* Leiden Algorithm — Community Detection
* FastAPI — GraphRAG Microservice (Python)

### 🌐 Live Intelligence & Data
* Tavily Search & Scraping
* MarketPrice Data Source

### ☁️ Communication Layer
* Twilio (Voice Calls)
* WhatsApp Integration
* SMS Notifications

### 🖼️ Multimodal Processing
* Image Query Understanding
* Visual Reasoning Pipeline

### ⚙️ Backend
* Node.js — AI Agent Core & API
* Python FastAPI — GraphRAG Microservice
* Session-based Conversation History