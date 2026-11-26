import os

import requests
import streamlit as st

BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000")

st.set_page_config(page_title="Document RAG Chat", page_icon="📄")
st.title("📄 Document RAG Assistant")

if "session_id" not in st.session_state:
    st.session_state.session_id = None
if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.header("Upload Document")
    uploaded_file = st.file_uploader(
        "Choose a PDF or XLSX file", type=["pdf", "xlsx"]
    )

    if uploaded_file and st.button("Process Document"):
        with st.spinner("Parsing and indexing..."):
            files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
            data = {}
            if st.session_state.session_id:
                data["session_id"] = st.session_state.session_id

            try:
                resp = requests.post(
                    f"{BACKEND_URL}/upload", files=files, data=data, timeout=120
                )
            except requests.RequestException as e:
                st.error(f"Request failed: {e}")
                resp = None

            if resp is not None:
                if resp.status_code == 200:
                    result = resp.json()
                    st.session_state.session_id = result["session_id"]
                    st.success(f"Indexed {result['chunks_count']} chunks!")
                else:
                    st.error(f"Upload failed: {resp.text}")

    if st.button("Clear Session"):
        if st.session_state.session_id:
            try:
                requests.delete(
                    f"{BACKEND_URL}/session/{st.session_state.session_id}",
                    timeout=30,
                )
            except requests.RequestException:
                pass
        st.session_state.session_id = None
        st.session_state.messages = []
        st.rerun()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if prompt := st.chat_input("Ask a question about your document..."):
    if not st.session_state.session_id:
        st.warning("Please upload a document first.")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    resp = requests.post(
                        f"{BACKEND_URL}/query",
                        data={
                            "session_id": st.session_state.session_id,
                            "question": prompt,
                        },
                        timeout=180,
                    )
                except requests.RequestException as e:
                    st.error(f"Request failed: {e}")
                    resp = None

                if resp is not None:
                    if resp.status_code == 200:
                        answer = resp.json()["answer"]
                        st.write(answer)
                        st.session_state.messages.append(
                            {"role": "assistant", "content": answer}
                        )
                    else:
                        st.error(f"Query failed: {resp.text}")