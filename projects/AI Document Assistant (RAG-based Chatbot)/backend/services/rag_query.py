from typing import List, Dict
from embeddings import load_embeddings, get_embedding_safe
import numpy as np
from google import genai
import os
from dotenv import load_dotenv

load_dotenv()
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

# -------------------- Similarity Function -------------------- #
def cosine_similarity(a: List[float], b: List[float]) -> float:
    a, b = np.array(a), np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


# -------------------- Retrieval -------------------- #
def retrieve_top_k(query: str, embeddings_data: List[Dict], k: int = 3) -> List[Dict]:
    query_emb = get_embedding_safe(query)

    if not query_emb:
        return []

    scored_chunks = []

    for item in embeddings_data:
        score = cosine_similarity(query_emb, item["embedding"])
        scored_chunks.append({
            "text": item["text"],
            "score": score,
            "source": item.get("source", "unknown")
        })

    # Sort by similarity score (descending)
    scored_chunks.sort(key=lambda x: x["score"], reverse=True)

    return scored_chunks[:k]


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

Use ONLY the provided context. Use conversation history only to understand references like "it", "he", or "that".

IMPORTANT:
- Identify which topic the user is referring to
- Do NOT mix information from different documents
- If the question refers to a different topic, switch context accordingly

Rules:
- Do NOT guess
- Do NOT add external knowledge
- Do NOT include XML, JSON, tags, or formatting
- Do NOT show reasoning or scratchpad
- Answer in a natural, conversational tone
- If answer is not clearly in context, say:
  "I couldn’t find that information in the provided documents."

Context:
{context_text}

History:
{history_text}

Question:
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
def rag_pipeline(query: str, embeddings_data: List[Dict], history: List[Dict] = None) -> Dict:
    if not embeddings_data:
        return {"answer": "No embeddings found", "sources": []}

    # Step 1: Retrieve
    top_chunks = retrieve_top_k(query, embeddings_data, k=3)
    # ✅ ADD THIS HERE
    if not top_chunks:
        return {
            "answer": "Answer not found in provided documents",
            "sources": []
        }

    # Step 2: Generate
    answer = generate_answer(query, top_chunks, history)

    # Step 3: Sources
    sources = [{
        "text": c["text"][:200],
        "score": c["score"],
        "source": c["source"]
    } for c in top_chunks]

    return {
        "answer": answer,
        "sources": sources
    }

# -------------------- TESTING BLOCK -------------------- #
if __name__ == "__main__":
    from pathlib import Path

    project_root = Path(__file__).parent.parent.parent
    embeddings_folder = project_root / "data" / "embeddings"

    all_embeddings = []
    loaded_files = []

    # Load all embeddings once
    for file in embeddings_folder.glob("*.pkl"):
        data = load_embeddings(file)
        all_embeddings.extend(data)
        loaded_files.append(file.stem)

    if not all_embeddings:
        print("No embeddings found. Please generate embeddings first.")
        exit()

    print(f"\nTotal documents loaded: {len(loaded_files)}")
    print("Documents:", loaded_files)

    # Chat memory
    chat_history = []

    try:
        while True:
            query = input("\nAsk a question (or 'exit'): ").lower().strip()

            if query.lower() == "exit":
                break

            # Optional: handle document listing
            if "documents" in query.lower() or "files" in query.lower():
                print("\nAvailable documents:")
                for doc in loaded_files:
                    print("-", doc)
                continue

            result = rag_pipeline(query, all_embeddings, chat_history)

            print("\nAnswer:\n")
            print(result["answer"])

            # print("\nSources:\n")
            # for i, s in enumerate(result["sources"]):
            #     print(f"Rank {i+1} | Score: {s['score']:.4f}")
            #     print(f"Source: {s['source']}")
            #     print(s["text"])
            #     print("-" * 40)

            # Save memory
            chat_history.append({
                "user": query,
                "assistant": result["answer"]
            })

    except Exception as e:
        print(f"Error: {e}")


