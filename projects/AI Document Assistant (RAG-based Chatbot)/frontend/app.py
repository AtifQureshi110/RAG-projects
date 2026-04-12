import streamlit as st
import requests
import re

API_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="DocuMind AI",
    layout="wide"
)

# ---------------- HEADER ---------------- #
st.title("DocuMind AI Assistant")

st.markdown(
    """
This system allows you to upload multiple document formats including PDF, DOCX, and TXT.
You can then ask questions and get answers directly from your documents using AI-powered retrieval.
"""
)

st.divider()

# ---------------- LAYOUT ---------------- #
col1, col2 = st.columns([1, 2])

# ================= LEFT PANEL: UPLOAD ================= #
with col1:
    st.subheader("Document Upload")

    st.markdown("Supported formats: PDF, DOCX, TXT")

    uploaded_file = st.file_uploader(
        "Upload your document",
        type=["pdf", "docx", "txt"]
    )

    if st.button("Upload Document", use_container_width=True):
        if uploaded_file is not None:
            try:
                files = {
                    "file": (
                        uploaded_file.name,
                        uploaded_file,
                        uploaded_file.type
                    )
                }

                response = requests.post(
                    f"{API_URL}/upload",
                    files=files
                )

                if response.status_code == 200:
                    st.success("Document uploaded successfully")
                    st.json(response.json())
                else:
                    st.error("Upload failed")
                    st.write(response.text)

            except Exception as e:
                st.error(f"Error: {str(e)}")
        else:
            st.warning("Please select a file before uploading")

# ---------------- DOCUMENT LIST ---------------- #
st.divider()
st.subheader("Available Documents")

# STEP 1: Load data
if st.button("Show Documents"):
    response = requests.get(f"{API_URL}/documents")
    st.session_state["docs_data"] = response.json()

# STEP 2: Always render if data exists
if "docs_data" in st.session_state:

    data = st.session_state["docs_data"]

    if "documents" in data and "count" in data:

        st.write(f"Total Documents: {data['count']}")
        st.markdown("### Documents")

        for i, doc in enumerate(data["documents"]):

            col_a, col_b = st.columns([4, 1])

            with col_a:
                st.markdown(f"- {doc}")

            with col_b:
                if st.button("Delete", key=f"delete_{i}"):

                    response = requests.delete(
                        f"{API_URL}/documents/{doc}"
                    )

                    result = response.json()

                    if "error" in result:
                        st.error(result["error"])
                    else:
                        st.success(result["message"])

                        # refresh data
                        del st.session_state["docs_data"]
                        st.rerun()
# ================= RIGHT PANEL: CHAT ================= #
with col2:
    st.subheader("Ask Questions from Documents")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    query = st.text_input("Enter your question")

    col_a, col_b = st.columns([1, 1])

    with col_a:
        ask_btn = st.button("Ask Question", use_container_width=True)

    with col_b:
        clear_btn = st.button("Clear Chat", use_container_width=True)

    if clear_btn:
        st.session_state.chat_history = []

    if ask_btn and query:
        try:
            response = requests.post(
                f"{API_URL}/query",
                json={"query": query}
            )

            result = response.json()

            st.session_state.chat_history.append({
                "question": query,
                "answer": result.get("answer", "No response")
            })

        except Exception as e:
            st.error(f"Request failed: {str(e)}")

    st.divider()

    # ---------------- CHAT DISPLAY ---------------- #
    for chat in reversed(st.session_state.chat_history):
        st.markdown("**Question:**")
        st.write(chat["question"])

        st.markdown("**Answer:**")
        st.write(chat["answer"])

        st.divider()

