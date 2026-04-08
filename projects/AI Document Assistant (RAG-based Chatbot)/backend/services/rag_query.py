import os, math, time
from typing import List, Dict
from dotenv import load_dotenv
from google import genai
from embeddings import load_embeddings, get_embedding_safe

# -------------------- Setup -------------------- #
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")

if not API_KEY:
    raise ValueError("GOOGLE_API_KEY not found")

client = genai.Client(api_key=API_KEY)

# -------------------- Cosine Similarity -------------------- #
def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    norm_a = math.sqrt(sum(a * a for a in vec1))
    norm_b = math.sqrt(sum(b * b for b in vec2))

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot_product / (norm_a * norm_b)

# -------------------- Retrieve Top-K -------------------- #
def retrieve_top_k(query: str, embeddings_data: List[Dict], k: int = 3) -> List[Dict]:
    query_embedding = get_embedding_safe(query)

    if query_embedding is None:
        return []

    scored_chunks = []

    for item in embeddings_data:
        score = cosine_similarity(query_embedding, item["embedding"])
        scored_chunks.append({
            "text": item["text"],
            "score": score
        })

    scored_chunks.sort(key=lambda x: x["score"], reverse=True)
    return scored_chunks[:k]

# -------------------- Build Prompt -------------------- #
def build_prompt(query: str, context_chunks: List[Dict], history: List[Dict] = None) -> str:
    """
    Create prompt with:
    - retrieved context
    - optional chat history
    """

    context_text = "\n\n".join(
        [f"[Chunk {i+1}]: {c['text']}" for i, c in enumerate(context_chunks)]
    )

    history_text = ""
    if history:
        for h in history[-3:]:  # last 3 turns only
            history_text += f"User: {h['user']}\nAssistant: {h['assistant']}\n"

    prompt = f"""
You are a helpful AI assistant.

Answer the question based on:
1. Provided context
2. Conversation history (if relevant)

Rules:
- Prefer exact information from context
- If partially available, you may infer carefully BUT do not guess names, facts, or roles
- Do NOT introduce external knowledge
- If answer is not clearly supported, say:
  "Answer not found in provided documents"

Conversation History:
{history_text}

Context:
{context_text}

Question:
{query}

Answer in a clear and human-friendly way:
"""

    return prompt

# -------------------- Generate Answer -------------------- #


def generate_answer(query: str, retrieved_chunks: List[Dict], history: List[Dict] = None) -> str:
    prompt = build_prompt(query, retrieved_chunks, history)

    retries = 3
    for attempt in range(retries):
        try:
            response = client.models.generate_content(
                model="models/gemini-2.5-flash",
                contents=prompt
            )
            return response.text

        except Exception as e:
            print(f"Attempt {attempt+1} failed: {e}")
            time.sleep(2)

    return "Model is currently busy. Please try again."
    
# -------------------- Main RAG Pipeline -------------------- #
def rag_pipeline(query: str, embeddings_path: str, history: List[Dict] = None) -> Dict:
    """
    Full RAG flow:
    query → retrieve → generate → return answer + sources
    """
    embeddings_data = load_embeddings(embeddings_path)

    if not embeddings_data:
        return {"answer": "No embeddings found", "sources": []}

    # Step 1: Retrieve
    top_chunks = retrieve_top_k(query, embeddings_data, k=3)
    # DEBUG: show what model is actually reading

    # uncommit this if you want to see the chuck 

    # print("\nTop Retrieved Chunk:\n")
    # if top_chunks:
    #     print(top_chunks[0]["text"][:500])
    # else:
    #     print("No chunks retrieved")

    # Step 2: Generate answer
    answer = generate_answer(query, top_chunks, history)

    # Step 3: Prepare sources
    sources = []
    for c in top_chunks:
        sources.append({
            "text": c["text"][:200],
            "score": c["score"]
        })

    return {
        "answer": answer,
        "sources": sources
    }

# -------------------- TESTING BLOCK -------------------- #
if __name__ == "__main__":
    from pathlib import Path

    project_root = Path(__file__).parent.parent.parent
    embeddings_path = project_root / "data" / "embeddings" / "kamran_taj_embeddings.pkl"

    # simple memory
    chat_history = []

    try:
        while True:
            query = input("\nAsk a question (or 'exit'): ")

            if query.lower() == "exit":
                break

            result = rag_pipeline(query, embeddings_path, chat_history)

            print("\nAnswer:\n")
            print(result["answer"])

            print("\nSources:\n")
            # for i, s in enumerate(result["sources"]):
            #     print(f"Rank {i+1} | Score: {s['score']:.4f}")
            #     print(s["text"])
            #     print("-" * 40)

            # update memory
            chat_history.append({
                "user": query,
                "assistant": result["answer"]
            })

    except Exception as e:
        print(f"Error: {e}")