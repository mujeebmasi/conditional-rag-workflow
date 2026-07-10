import os
from dotenv import load_dotenv
from typing import TypedDict, Annotated
#annotated is used to provide type hints for the input and output of the function
#annotated is used to store multiple things in a single data structure(dict)
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings, embeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

#STEP 1 -> BUILDING THE RAG RETRIEVERS.
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def build_retriever(pdf_path: str):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_documents(documents)
    vectorstore = FAISS.from_documents(chunks, embeddings) #here chunks are converted to vectors and stored in FAISS
    
    return vectorstore.as_retriever(search_kwargs = {"k" : 4 }) 

academic_retriever = build_retriever("data/academic.pdf")
fee_retriever = build_retriever("data/fee_structure.pdf")

llm = ChatGroq(model = "llama-3.3-70b-versatile", temperature=0.4)

#STEP 2 -> BUILDING STATEE

class State(TypedDict):
    program : str
    messages : Annotated[list, add_messages]
    query_type : str
    retrieved_context : str
    inner_state : str


