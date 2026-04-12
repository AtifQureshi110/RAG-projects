from pathlib import Path
from docx import Document
from pypdf import PdfReader
import re
from typing import List

# Load_document
def load_document(file_path: str) -> str: #type hints improve readability
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"{file_path} not found")

    if file_path.suffix.lower() == ".txt":
        return file_path.read_text(encoding="utf-8")

    elif file_path.suffix.lower() == ".pdf":
        reader = PdfReader(file_path)
        texts = []
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                texts.append(page_text.strip()) # removes extra leading/trailing spaces or newlines.
        return "\n".join(texts)

    elif file_path.suffix.lower() == ".docx":
        doc = Document(file_path)
        return "\n".join(
            p.text.strip() for p in doc.paragraphs if p.text.strip()
        )
    else:
        raise ValueError(f"Unsupported file type: {file_path.suffix}")

# Clean_text
def clean_text(text: str) -> str:
    """
    Clean extracted text for RAG:
    - Normalize whitespace
    - Fix broken words from PDF extraction
    - Remove excessive newlines
    - Keep semantic structure intact
    """

    if not text:
        return ""

    # 1. Normalize line breaks
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # 2. Remove excessive newlines (keep paragraph structure)
    text = re.sub(r"\n{3,}", "\n\n", text)

    # 3. Fix broken words (very common in PDFs: "h is" -> "his")
    text = re.sub(r"\b(\w)\s+(\w)\b", r"\1\2", text)

    # 4. Remove extra spaces
    text = re.sub(r"[ \t]+", " ", text)

    # 5. Trim spaces around newlines
    text = re.sub(r" *\n *", "\n", text)

    # 6. Strip overall text
    text = text.strip()

    return text

# split_text
def split_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """
    Split text into chunks for RAG:
    - Tries to keep semantic boundaries (paragraphs first)
    - Falls back to sentence splitting if needed
    - Adds overlap between chunks
    """

    if not text:
        return []

    #Split into paragraphs
    paragraphs = text.split("\n\n")

    chunks = []
    current_chunk = []

    def word_count(t):
        return len(t.split())

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        # If paragraph itself is too big → split into sentences
        if word_count(para) > chunk_size:
            sentences = re.split(r'(?<=[.!?]) +', para)
        else:
            sentences = [para]

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            current_chunk.append(sentence)

            if word_count(" ".join(current_chunk)) >= chunk_size:
                chunk_text = " ".join(current_chunk)
                chunks.append(chunk_text)

                # Add overlap -->Overlap prevents loss of context between chunks, which improves retrieval quality in RAG systems.
                overlap_words = chunk_text.split()[-overlap:]
                current_chunk = [" ".join(overlap_words)]

    # Add remaining chunk
    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

def process_document(file_path: str) -> List[str]:
    """
    Full RAG pipeline:
    load → clean → split
    """
    raw_text = load_document(file_path)
    cleaned_text = clean_text(raw_text)
    chunks = split_text(cleaned_text)
    return chunks

# # ================== TESTING BLOCK ==================
# if __name__ == "__main__":
#     from pathlib import Path

#     project_root = Path(__file__).parent.parent.parent
#     data_folder = project_root / "data" / "uploaded_docs"

#     file_path = data_folder / "refference letter_Atif_DR. ARIFA BHUTTO, PHD.pdf"

#     try:
#         # Debug mode (step-by-step inspection)
#         raw_text = load_document(file_path)
#         cleaned_text = clean_text(raw_text)
#         chunks = split_text(cleaned_text)

#         print("RAW PREVIEW:\n", raw_text[:500])
#         print("\n" + "="*50 + "\n")
#         print("CLEANED PREVIEW:\n", cleaned_text[:500])

#         print(f"\nTotal chunks: {len(chunks)}\n")

#         for i, chunk in enumerate(chunks[:3]):
#             print(f"--- Chunk {i+1} ---")
#             print(chunk[:300])
#             print()

#         # Pipeline mode (final check)
#         pipeline_chunks = process_document(file_path)

#         print("\n" + "="*50)
#         print("PIPELINE OUTPUT CHECK")
#         print(f"Chunks from process_document: {len(pipeline_chunks)}")

#     except Exception as e:
#         print(f"Error: {e}")