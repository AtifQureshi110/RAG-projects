# embeddings.py
import os, time
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
        #gemini-3.1-flash-lite-preview + gemini-embedding-001
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

# ================== TESTING BLOCK ==================


# from pathlib import Path
# from backend.services.document_processing import load_document, clean_text, split_text
# from pinecone_utils import upload_embeddings


# if __name__ == "__main__":

#     project_root = Path(__file__).parent.parent.parent
#     data_folder = project_root / "data" / "uploaded_docs"

#     # GET ALL FILES
#     files = list(data_folder.glob("*"))

#     print(f"Found {len(files)} documents")

#     for file_path in files:
#         try:
#             print(f"\nProcessing: {file_path.name}")

#             # ---------------- LOAD ---------------- #
#             raw_text = load_document(file_path)
#             cleaned_text = clean_text(raw_text)
#             chunks = split_text(cleaned_text)

#             print(f"Total chunks received: {len(chunks)}")

#             # ---------------- EMBEDDINGS ---------------- #
#             embeddings_data = get_embeddings(chunks)

#             print(f"Embeddings generated: {len(embeddings_data)}")

#             # ---------------- UPLOAD TO PINECONE ---------------- #
#             file_name = file_path.stem

#             upload_embeddings(
#                 embeddings_data=embeddings_data,
#                 document_name=file_name
#             )

#             print(f"Uploaded: {file_name}")

#         except Exception as e:
#             print(f"Error processing {file_path.name}: {e}")