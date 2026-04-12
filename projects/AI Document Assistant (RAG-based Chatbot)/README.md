# SmartDoc AI Assistant (RAG-Based Document Q&A System)

## Overview
SmartDoc AI Assistant is a Retrieval-Augmented Generation (RAG) system that allows users to upload documents (PDF, TXT, DOCX), process them into embeddings, and ask questions in natural language.

The system retrieves relevant context using vector search and generates accurate answers using an LLM.

---

## Features
- Upload documents (PDF, TXT, DOCX)
- Automatic text chunking with overlap
- Embedding generation using LLM
- Vector search using Pinecone
- AI-powered question answering
- Document management (list & delete)
- Chat history memory
- Streamlit UI + FastAPI backend

---

## Tech Stack
- FastAPI (Backend)
- Streamlit (Frontend)
- Pinecone (Vector DB)
- Google Gemini API (LLM)
- Python

---

## System Architecture
                                               +--------------------------+
                            |      USER (Browser)      |
                            |   Streamlit Frontend     |
                            +------------+-------------+
                                         |
                                         | HTTP Requests
                                         v
                            +--------------------------+
                            |   FASTAPI BACKEND API    |
                            |   (Core Orchestrator)    |
                            +------------+-------------+
                                         |
         +-------------------------------+-------------------------------+
         |                               |                               |
         v                               v                               v
+------------------+      +--------------------------+     +------------------------+
| Upload Service   |      | Query / RAG Engine       |     | Document Manager      |
|                  |      |                          |     |                        |
| File Storage     |      | Retrieval System        |     | List / Delete API     |
+--------+---------+      +------------+-------------+     +-----------+------------+
         |                             |                               |
         +-------------+---------------+---------------+---------------+
                       |
                       v
        +------------------------------------------------------+
        |           DOCUMENT PROCESSING PIPELINE              |
        |                                                      |
        |  PDF / TXT / DOCX                                   |
        |        ↓                                             |
        |  Text Extraction                                    |
        |        ↓                                             |
        |  Chunking (with overlap)                            |
        +---------------------+--------------------------------+
                              |
                              v
        +------------------------------------------------------+
        |        EMBEDDING SERVICE (Gemini API)               |
        |   Text Chunks → Vector Embeddings (3072-d)          |
        +---------------------+--------------------------------+
                              |
                              v
        +------------------------------------------------------+
        |           PINECONE VECTOR DATABASE                  |
        |  Stores vectors + metadata (doc_name, doc_id)      |
        +---------------------+--------------------------------+
                              |
                              v
        +------------------------------------------------------+
        |             RETRIEVAL ENGINE (RAG)                  |
        |   Query → Embedding → Top-K Similar Chunks          |
        +---------------------+--------------------------------+
                              |
                              v
        +------------------------------------------------------+
        |            LLM GENERATION (Gemini)                  |
        |   Context + History → Prompt → Final Answer         |
        +---------------------+--------------------------------+
                              |
                              v
                     +----------------------+
                     |  RESPONSE TO USER    |
                     +----------------------+

---

## How It Works
1. User uploads document
2. Document is chunked
3. Embeddings are generated
4. Stored in Pinecone
5. User asks question
6. Query is embedded
7. Similar chunks retrieved
8. LLM generates final answer

---

## API Endpoints
- POST /upload → Upload document
- POST /query → Ask question
- GET /documents → List documents
- DELETE /documents/{doc} → Delete document

---

## Screenshots
data/images and videos/UI_01.jpg
data/images and videos/UI_02.jpg
data/images and videos/UI_03.jpg

---

## Setup Instructions
```bash
# 1. Clone the repository
git clone https://github.com/AtifQureshi110/RAG-projects.git

# 2. Navigate to project folder
cd "RAG-projects/projects/AI Document Assistant (RAG-based Chatbot)"

# 3. Create virtual environment (optional but recommended)
python -m venv .venv

# 4. Activate virtual environment
# Windows:
.venv\Scripts\activate

# 5. Install dependencies
pip install -r requirements.txt

# 6. Run FastAPI backend
uvicorn backend.main:app --reload

# 7. Run Streamlit frontend (open new terminal)
streamlit run frontend/app.py