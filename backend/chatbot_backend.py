from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    CSVLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredURLLoader
)
from langchain.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

import os

def load_documents(source, file_type):
    if file_type == "pdf":
        loader = PyPDFLoader(source)
    elif file_type == "txt":
        loader = TextLoader(source)
    elif file_type == "csv":
        loader = CSVLoader(source)
    elif file_type == "docx":
        loader = UnstructuredWordDocumentLoader(source)
    elif file_type == "url":
        loader = UnstructuredURLLoader(urls=[source])
    else:
        raise ValueError("Unsupported file type.")
    
    return loader.load()

def build_vectorstore(docs):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(texts, embeddings)
    return vectorstore

def get_qa_chain(vectorstore):
    llm = ChatOpenAI(temperature=0)
    chain = RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever())
    return chain
