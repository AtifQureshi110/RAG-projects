# embeddings.py
import os, pickle, time
from typing import List, Dict, Union
from dotenv import load_dotenv
from google import genai

# -------------------- ENV & CLIENT INIT -------------------- #
load_dotenv()  # Load variables from .env
API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in .env")

# Initialize Gemini client
client = genai.Client(api_key=API_KEY)

# -------------------- Single Text Embedding -------------------- #
def get_embedding(text: str) -> Union[List[float], None]:
    """
    Generate embedding for a single text chunk using Gemini.
    """
    if not text or not text.strip():
        return None

    try:
        response = client.models.embed_content(
            model="models/gemini-embedding-001",
            contents=text
        )
        # Extract embedding vector
        emb_vector = response.embeddings[0].values
        return emb_vector

    except Exception as e:
        print(f"Error generating embedding: {e}")
        return None

# -------------------- Safe Retry Wrapper -------------------- #
def get_embedding_safe(text: str, retries: int = 3, delay: float = 2.0) -> Union[List[float], None]:
    """
    Retry wrapper for embedding to avoid API freeze / failure.
    """
    for i in range(retries):
        emb = get_embedding(text)
        if emb:
            return emb
        print(f"Retry {i+1}/{retries} for embedding...")
        time.sleep(delay)
    print("Failed to generate embedding after retries")
    return None

# -------------------- List of Chunks Embeddings -------------------- #
def get_embeddings(chunks: List[str]) -> List[Dict]:
    """
    Generate embeddings for a list of text chunks.
    Returns list of dicts: {"text": chunk, "embedding": emb_vector}
    """
    results = []
    for i, chunk in enumerate(chunks):
        if isinstance(chunk, list):
            chunk = " ".join(chunk)  # flatten
        emb = get_embedding_safe(chunk)
        if emb:
            results.append({"text": chunk, "embedding": emb})
        else:
            print(f"Skipped chunk {i} due to empty or failed embedding")
    return results

# -------------------- Save / Load Embeddings -------------------- #
def save_embeddings(file_path: str, embeddings: List[Dict]) -> None:
    """Save embeddings to disk using pickle."""
    try:
        with open(file_path, "wb") as f:
            pickle.dump(embeddings, f)
        print(f"✅ Embeddings saved successfully at: {file_path}")
    except Exception as e:
        print(f"Error saving embeddings: {e}")


def load_embeddings(file_path: str) -> List[Dict]:
    """Load embeddings previously saved on disk."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Embeddings file not found: {file_path}")
    with open(file_path, "rb") as f:
        embeddings = pickle.load(f)
    print(f"✅ Embeddings loaded successfully from: {file_path}")
    return embeddings

# ================== TESTING BLOCK ==================
if __name__ == "__main__":
    from pathlib import Path
    from document_processing import load_document, clean_text, split_text

    project_root = Path(__file__).parent.parent.parent
    data_folder = project_root / "data" / "uploaded_docs"
    file_path = data_folder / "kamran_taj.pdf"

    try:
        raw_text = load_document(file_path)
        cleaned_text = clean_text(raw_text)
        chunks = split_text(cleaned_text)
        print(f"Total chunks received: {len(chunks)}")

        embeddings_data = get_embeddings(chunks)
        print(f"Generated embeddings for {len(embeddings_data)} chunks")

        if embeddings_data:
            first_emb = embeddings_data[0]["embedding"]
            print(f"Length of first embedding vector: {len(first_emb)}")
            print(f"Preview of first embedding (first 10 values): {first_emb[:10]}")

        # Save & Load
        save_path = project_root / "data" / "embeddings" / "kamran_taj_embeddings.pkl"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_embeddings(save_path, embeddings_data)

        loaded_embeddings = load_embeddings(save_path)
        print(f"Loaded embeddings count: {len(loaded_embeddings)}")
        if loaded_embeddings:
            print("Preview loaded text:", loaded_embeddings[0]["text"][:200])
            print("Embedding length:", len(loaded_embeddings[0]["embedding"]))

    except Exception as e:
        print(f"Error in embeddings pipeline: {e}")