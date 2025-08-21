# 🤖 Agentic RAG with LangGraph & Groq LLM

An **Agentic Retrieval-Augmented Generation (RAG)** system built using **LangGraph, LangChain, FAISS, HuggingFace Embeddings, and Groq’s LLaMA-3.3 70B LLM**.  

This project demonstrates how to combine **multi-step agent reasoning, retrieval tools, and query rewriting** into a single workflow that can **dynamically decide when to retrieve, rewrite, or generate answers** — a next-gen intelligent retrieval system.

## 🚀 Project Overview

Traditional RAG pipelines retrieve documents and generate answers directly. However, they often fail if the retrieved documents are irrelevant or if the query is poorly phrased.
This project enhances the traditional RAG pipeline by making it **agentic** – that means it can **think, decide, and adapt** at every step. Instead of just blindly retrieving and generating, the workflow is **dynamic and self-correcting**.

## Here’s how it works:
1. **Agent (Decision Maker)**  
   - The agent receives the user’s query.  
   - It decides whether the query needs additional context (retrieval) or if it can be answered directly.  
2. **Retriever (Knowledge Seeker)**  
   - If retrieval is needed, the agent calls specialized retrievers (vector stores).  
   - These retrievers fetch potentially relevant documents from **LangGraph** and **LangChain** tutorials.  
3. **Relevance Grader (Quality Inspector)**  
   - Every retrieved document is **evaluated for relevance**.  
   - The grader acts like a filter:  
     - ✅ If documents are **relevant**, the process moves forward.  
     - ❌ If documents are **irrelevant**, the system does not waste time — it triggers query refinement.  
4. **Rewriter (Query Improver)**  
   - When documents don’t match well, the query is **rewritten intelligently**.  
   - The goal is to capture the **semantic intent** behind the user’s question, not just the literal words.  
   - The refined query is then sent back to the Agent, creating a feedback loop.  
5. **Generator (Answer Creator)**  
   - Once relevant documents are confirmed, the generator combines them with the query to **produce a final, well-grounded answer**.  
   - The generator is powered by **Groq’s LLaMA-3.3 70B LLM**, ensuring high-quality, context-aware responses.  
This way, the system mimics how a **human researcher** thinks: *ask → search → validate → refine → answer*.


## 🔄 Self-Correcting Feedback Loop
Unlike static RAG, this workflow is **iterative and adaptive**.  
If the first attempt fails, it doesn’t just give up or hallucinate. Instead, it:
- **Checks relevance**  
- **Rewrites the query if needed**  
- **Re-queries the retriever**  
- **Repeats until a strong answer is produced**  

This process mimics how a **skilled human researcher** operates:
1. Ask a question  
2. Search for information  
3. Judge the quality of results  
4. Refine the question if needed  
5. Formulate a reliable answer 

By integrating **decision-making, validation, and refinement** into the RAG pipeline, this project delivers:  
- **Higher accuracy** → Answers are based on relevant documents only.  
- **Robustness** → Poorly phrased queries are auto-corrected.  
- **Human-like reasoning** → The system actively learns and adapts during the conversation.  

In short, **Agentic RAG transforms a passive retrieval system into an intelligent, autonomous assistant**.

## 🏗️ Architecture & Workflow

flowchart TD
    A [User Query]     --> B[Agent]
    B --> Use Tool     --> C[Retriever]
    B --> No Tool      --> G[End]
    C --> D[Relevance Grader]
    D --> Relevant     --> E[Generate Answer]
    D --> Not Relevant --> F[Rewrite Query]
    F --> B
    E --> G[End]

## 🏗️ Architecture & Workflow

The Agentic RAG workflow is captured below:

![Agentic RAG Workflow](./output.png)

**Key Nodes:**
- **Agent** → Decides whether to retrieve or not.  
- **Retriever** → Fetches documents from FAISS vector stores.  
- **Relevance Grader** → Validates if docs are useful.  
- **Rewriter** → Reframes poor queries.  
- **Generator** → Produces the final answer using Groq LLaMA-3.3 70B.  

## 🔑 Key Features
- ⚡ Multi-Agent Reasoning: Agent can decide when to use tools or finish.
- 📚 Two Knowledge Sources:
- LangGraph tutorials
- LangChain tutorials
- 🧠 Intelligent Grading: Checks document relevance before answering.
- 🔄 Self-Correcting: Automatically rewrites poor queries.
- 🚀 Powered by Groq LLM: Ultra-fast inference with llama-3.3-70b-versatile.
- 🔍 FAISS + HuggingFace MiniLM embeddings for semantic retrieval.

## 🛠️ Tech Stack
- LangGraph → Workflow orchestration for agentic RAG.
- LangChain → Tools, retrievers, and utilities.
- Groq LLM (llama-3.3-70b-versatile) → Reasoning & generation.
- FAISS → Vector database for similarity search.
- HuggingFace Sentence Transformers (all-MiniLM-L6-v2) → Embeddings.
- dotenv → Environment variable management.
- Python → Core implementation.

## Run the pipeline:
- response = graph.invoke({"messages": "What is the use of LangChain and what are the components of langchain?"})

## The agent will:
- Retrieve relevant docs,
- Grade them,
- Rewrite if needed,
- And finally generate an answer.

## 💡 Why This Project Matters
This project showcases the ability to design next-gen AI pipelines that go beyond simple RAG. By combining multi-agent workflows, query rewriting, and validation, it demonstrates strong skills in:
- AI Agent Orchestration
- LLM Tool Integration
- Vector Search & Semantic Retrieval
- Production-Ready RAG Design

