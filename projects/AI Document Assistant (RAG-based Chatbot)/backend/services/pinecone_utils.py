import os
import hashlib
from typing import List, Dict
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
import hashlib


# -------------------- ENV SETUP -------------------- #
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY not found in environment variables")

# Init Pinecone client
pc = Pinecone(api_key=PINECONE_API_KEY)

# Index name
INDEX_NAME = "ai-multi-document-assistant-chatbot"


# -------------------- HELPERS -------------------- #

def get_dimension(embeddings_data: List[Dict]) -> int:
    """Dynamically detect embedding dimension."""
    if not embeddings_data:
        raise ValueError("Embeddings data is empty")

    return len(embeddings_data[0]["embedding"])


def generate_doc_id(document_name: str) -> str:
    """Generate unique hash for document."""
    return hashlib.md5(document_name.encode()).hexdigest()


# -------------------- INDEX MANAGEMENT -------------------- #

def get_index(dimension: int):
    """Create index if not exists, otherwise connect."""
    existing_indexes = [i["name"] for i in pc.list_indexes()]

    if INDEX_NAME not in existing_indexes:
        print(f"Creating Pinecone index (dim={dimension})...")

        pc.create_index(
            name=INDEX_NAME,
            dimension=dimension,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )

        print("Index created successfully")

    return pc.Index(INDEX_NAME)

# -------------------- UPLOAD EMBEDDINGS -------------------- #
def get_chunk_hash(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()

def upload_embeddings(embeddings_data: List[Dict], document_name: str):

    if not embeddings_data:
        print("No embeddings to upload")
        return

    dimension = get_dimension(embeddings_data)
    index = get_index(dimension)

    doc_id = generate_doc_id(document_name)

    vectors = []

    for i, item in enumerate(embeddings_data):

        # create chunk-level hash INSIDE loop
        chunk_hash = get_chunk_hash(item["text"])

        vectors.append({
            "id": f"{doc_id}-{chunk_hash}",   # FIXED (content-based ID)
            "values": item["embedding"],
            "metadata": {
                "text": item["text"],
                "source": document_name,
                "doc_id": doc_id,
                "chunk_hash": chunk_hash   # (optional but recommended)
            }
        })

    batch_size = 100

    for i in range(0, len(vectors), batch_size):
        index.upsert(vectors=vectors[i:i + batch_size])

    print(f"Uploaded {len(vectors)} vectors for '{document_name}'")
# -------------------- OPTIONAL: DELETE DOCUMENT -------------------- #

def delete_document(document_name: str):
    """Delete all vectors of a specific document."""
    doc_id = generate_doc_id(document_name)
    index = pc.Index(INDEX_NAME)

    print(f"🗑 Deleting document: {document_name}")

    index.delete(filter={"doc_id": doc_id})

    print("Document deleted successfully")

# ================== TESTING BLOCK ==================

# if __name__ == "__main__":

#     from pathlib import Path
#     from backend.services.document_processing import load_document, clean_text, split_text
#     from backend.services.embeddings import get_embeddings

#     project_root = Path(__file__).parent.parent.parent
#     data_folder = project_root / "data" / "uploaded_docs"

#     files = list(data_folder.glob("*"))

#     print(f"\nFound {len(files)} documents\n")

#     for file_path in files:
#         try:
#             print(f"\nProcessing: {file_path.name}")

#             # ---------------- LOAD ---------------- #
#             raw_text = load_document(file_path)
#             cleaned_text = clean_text(raw_text)
#             chunks = split_text(cleaned_text)

#             print(f"Total chunks: {len(chunks)}")

#             # ---------------- EMBEDDINGS ---------------- #
#             embeddings_data = get_embeddings(chunks)

#             print(f"Embeddings generated: {len(embeddings_data)}")

#             # ---------------- UPLOAD ---------------- #
#             upload_embeddings(
#                 embeddings_data=embeddings_data,
#                 document_name=file_path.stem
#             )

#             print(f"Uploaded: {file_path.name}")

#         except Exception as e:
#             print(f"Error processing {file_path.name}: {e}")