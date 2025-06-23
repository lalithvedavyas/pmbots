from langchain.document_loaders import (
    PyPDFLoader, TextLoader, UnstructuredWordDocumentLoader,
    CSVLoader, UnstructuredExcelLoader, WebBaseLoader
)
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA

def load_documents(file_path, file_type):
    if file_type == "pdf":
        loader = PyPDFLoader(file_path)
    elif file_type == "txt":
        loader = TextLoader(file_path)
    elif file_type == "docx":
        loader = UnstructuredWordDocumentLoader(file_path)
    elif file_type == "csv":
        loader = CSVLoader(file_path)
    elif file_type == "xlsx":
        loader = UnstructuredExcelLoader(file_path)
    elif file_type == "url":
        loader = WebBaseLoader(file_path)
    else:
        raise ValueError("Unsupported format")
    return loader.load()

def build_vectorstore(docs):
    embeddings = OpenAIEmbeddings()
    return Chroma.from_documents(docs, embeddings)

def get_qa_chain(vectorstore):
    retriever = vectorstore.as_retriever()
    llm = OpenAI(temperature=0)
    return RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
