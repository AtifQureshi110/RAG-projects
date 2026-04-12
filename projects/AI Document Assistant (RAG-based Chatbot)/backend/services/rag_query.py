from typing import List, Dict
from backend.services.embeddings import get_embedding_safe
from backend.services.pinecone_utils import pc, INDEX_NAME
import numpy as np
from google import genai
import os
from dotenv import load_dotenv

load_dotenv()
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

# -------------------- Retrieval -------------------- #

def retrieve_top_k(query: str, k: int = 3):
    # 1. Convert query → embedding
    query_emb = get_embedding_safe(query)

    if not query_emb:
        return []

    # 2. Connect to index
    index = pc.Index(INDEX_NAME)

    # 3. Pinecone search (REAL VECTOR SEARCH)
    results = index.query(
        vector=query_emb,
        top_k=k,
        include_metadata=True
    )

    # 4. Format response
    scored_chunks = []

    for match in results["matches"]:
        scored_chunks.append({
            "text": match["metadata"]["text"],
            "score": match["score"],
            "source": match["metadata"].get("source", "unknown")
        })

    return scored_chunks

# -------------------- Generation -------------------- #
def generate_answer(query: str, chunks: List[Dict], history: List[Dict] = None) -> str:

    context_text = "\n\n".join([c["text"] for c in chunks])

    history_text = ""
    if history:
        history_text = "\n".join([
            f"User: {h['user']}\nAssistant: {h['assistant']}"
            for h in history[-3:]
        ])

    prompt = f"""
You are a helpful AI assistant.

You have TWO sources of information:
1. Conversation History (past chat)
2. Retrieved Documents

INSTRUCTIONS:
- If the question is about past conversation (e.g., "what did I ask before"), use ONLY the history.
- If the question is about documents, use ONLY the document context.
- Do NOT mix unrelated information.
- Decide intelligently which source to use.

Rules:
- Do NOT guess
- Do NOT add external knowledge
- Answer clearly and naturally
- If answer is not found, say:
  "I couldn’t find that information in the provided documents."

Conversation History:
{history_text}

Document Context:
{context_text}

User Question:
{query}

Answer:
"""
    for i in range(3):
        try:
            response = client.models.generate_content(
                model="models/gemini-2.5-flash",
                contents=prompt
            )
            return response.text.strip()

        except Exception as e:
            return f"Error generating answer: {e}"
    return "Temporary server issue, please try again"


# -------------------- RAG Pipeline -------------------- #
def rag_pipeline(query: str, history: list = None) -> dict:

    if history is None:
        history = []

    # ✅ memory fast path
    memory_keywords = ["last question", "before", "previous", "earlier", "repeat"]

    if any(k in query.lower() for k in memory_keywords):
        return {
            "answer": history[-1]["user"] if history else "No previous question found",
            "sources": []
        }

    # ✅ retrieval
    top_chunks = retrieve_top_k(query, k=5)
    top_chunks = [c for c in top_chunks if c["score"] > 0.5]

    print("QUERY:", query)
    print("CHUNKS:", len(top_chunks))

    if not top_chunks:
        return {
            "answer": "Answer not found in provided documents",
            "sources": []
        }

    # ✅ generation
    answer = generate_answer(query, top_chunks, history)

    # ✅ sources
    sources = [{
        "text": c["text"][:200],
        "score": c["score"],
        "source": c["source"]
    } for c in top_chunks]

    return {
        "answer": answer,
        "sources": sources
    }


# # -------------------- TESTING BLOCK (PINECONE VERSION) -------------------- #
# if __name__ == "__main__":
#     print("\nRAG SYSTEM TEST MODE (Pinecone)\n")

#     chat_history = []

#     try:
#         while True:
#             query = input("\nAsk a question (or type 'exit'): ").strip()

#             if query.lower() == "exit":
#                 print("\nExiting RAG test mode...\n")
#                 break

#             # Optional: quick document info test
#             if "documents" in query.lower() or "files" in query.lower():
#                 print("\nThis system uses Pinecone (no local file list available).")
#                 continue

#             # Run full RAG pipeline
#             result = rag_pipeline(query, chat_history)

#             print("\nANSWER:\n")
#             print(result["answer"])

#             print("\nSOURCES:")
#             for i, s in enumerate(result["sources"], 1):
#                 print(f"\n--- Source {i} ---")
#                 print(f"Score: {s['score']:.4f}")
#                 print(f"Source: {s['source']}")
#                 print(f"Text: {s['text']}")

#             # Save chat history
#             chat_history.append({
#                 "user": query,
#                 "assistant": result["answer"]
#             })

#     except Exception as e:
#         print(f"\nError occurred: {e}")