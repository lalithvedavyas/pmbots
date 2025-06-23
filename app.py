import streamlit as st
from backend.chatbot_backend import load_documents, build_vectorstore, get_qa_chain
import tempfile
import os

# Load OpenAI API key from secrets
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

st.title("📚 Smart Knowledge Base Chatbot")
st.write("Upload documents or provide a website URL. Then ask your questions!")

uploaded_file = st.file_uploader("Upload File", type=["pdf", "txt", "docx", "csv", "xlsx"])
url_input = st.text_input("Or enter a website URL")

if uploaded_file or url_input:
    with st.spinner("Processing..."):
        if uploaded_file:
            suffix = uploaded_file.name.split(".")[-1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{suffix}") as tmp:
                tmp.write(uploaded_file.read())
                docs = load_documents(tmp.name, suffix)
        elif url_input:
            docs = load_documents(url_input, "url")

        vectorstore = build_vectorstore(docs)
        qa_chain = get_qa_chain(vectorstore)
        st.session_state.qa_chain = qa_chain
        st.success("Knowledge base ready!")

if "qa_chain" in st.session_state:
    query = st.text_input("Ask your question:")
    if st.button("Submit"):
        with st.spinner("Generating answer..."):
            answer = st.session_state.qa_chain.run(query)
            st.write("💬 Answer:", answer)
