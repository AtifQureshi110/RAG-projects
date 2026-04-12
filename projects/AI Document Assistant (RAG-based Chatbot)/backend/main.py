from fastapi import FastAPI, UploadFile, File
from pathlib import Path
import shutil, uuid, re
from pydantic import BaseModel
from collections import defaultdict

from backend.services.document_processing import process_document
from backend.services.embeddings import get_embeddings
from backend.services.rag_query import rag_pipeline
from backend.services.pinecone_utils import upload_embeddings, pc, INDEX_NAME

app = FastAPI()

UPLOAD_DIR = Path("data/uploaded_docs")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

CHAT_HISTORY = []


@app.get("/")
def root():
    return {"message": "API is running"}

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):

    try:
        print("UPLOAD STARTED:", file.filename)

        file_path = UPLOAD_DIR / f"{uuid.uuid4()}_{file.filename}"

        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        print("FILE SAVED")

        chunks = process_document(str(file_path))
        print("CHUNKS:", len(chunks))

        embeddings_data = get_embeddings(chunks)
        print("EMBEDDINGS:", len(embeddings_data))

        upload_embeddings(
            embeddings_data=embeddings_data,
            document_name=file.filename
        )

        print("UPLOAD DONE")

        return {
            "message": "Document uploaded successfully",
            "chunks": len(chunks),
            "embedded": len(embeddings_data)
        }

    except Exception as e:
        print("ERROR OCCURRED:", str(e))
        return {"error": str(e)}

class QueryRequest(BaseModel):
    query: str


@app.post("/query")
def query_rag(request: QueryRequest):

    result = rag_pipeline(
        request.query,
        CHAT_HISTORY
    )

    CHAT_HISTORY.append({
        "user": request.query,
        "assistant": result["answer"]
    })

    return result

import re 
@app.get("/documents")
def list_documents():
    try:
        index = pc.Index(INDEX_NAME)

        stats = index.describe_index_stats()
        dimension = stats["dimension"]

        query_response = index.query(
            vector=[0] * dimension,
            top_k=10000,
            include_metadata=True
        )

        docs = set()

        for match in query_response["matches"]:
            meta = match.get("metadata", {})

            if "source" in meta:
                raw_name = meta["source"]

                clean_name = re.sub(
                    r'^[a-f0-9\-]{36}_',
                    '',
                    raw_name
                )

                docs.add(clean_name.strip())

        # ✅ RETURN OUTSIDE LOOP (IMPORTANT FIX)
        return {
            "documents": list(docs),
            "count": len(docs)
        }

    except Exception as e:
        return {"error": str(e)}


@app.delete("/documents/{document_name}")
def delete_document(document_name: str):
    try:
        index = pc.Index(INDEX_NAME)

        doc_id = None

        # 1. find vectors belonging to document
        query_response = index.query(
            vector=[0] * 3072,   # or dynamic dimension if you want
            top_k=10000,
            include_metadata=True
        )

        for match in query_response["matches"]:
            meta = match.get("metadata", {})

            if meta.get("source") == document_name:
                doc_id = meta.get("doc_id")
                break

        if not doc_id:
            return {"error": "Document not found"}

        # 2. delete using filter
        index.delete(
            filter={
                "doc_id": doc_id
            }
        )

        return {
            "message": f"Deleted document: {document_name}"
        }

    except Exception as e:
        return {"error": str(e)}