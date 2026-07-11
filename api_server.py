import os
from functools import lru_cache
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel

load_dotenv()


class State(TypedDict):
    programme: str
    messages: Annotated[list, add_messages]
    query_type: str
    retrieved_context: str


class ChatRequest(BaseModel):
    programme: str
    message: str


class ChatResponse(BaseModel):
    answer: str


@lru_cache(maxsize=1)
def get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


@lru_cache(maxsize=2)
def build_retriever(pdf_path: str):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_documents(documents)
    vectorstore = FAISS.from_documents(chunks, get_embeddings())
    return vectorstore.as_retriever(search_kwargs={"k": 4})


@lru_cache(maxsize=1)
def build_graph_app():
    academic_retriever = build_retriever("academics_handbook.pdf")
    fee_retriever = build_retriever("fee_structure.pdf")
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.4)

    def classifier_node(state: State) -> dict:
        last_message = state["messages"][-1].content
        prompt = (
            "Classify the following student query into exactly one category: "
            "'academic', 'fee', or 'general'.\n\n"
            "Use 'academic' for questions about attendance, exams, grading, credits, "
            "promotion, course structure, summer training, or degree requirements.\n"
            "Use 'fee' for questions about tuition, payment, refund, late charges, "
            "scholarships, or any money-related topic.\n"
            "Use 'general' for greetings, casual talk, or anything not related to "
            "the college rules or fee.\n\n"
            f"Query: {last_message}\n\n"
            "Return only one word: academic, fee, or general."
        )
        response = llm.invoke(prompt)
        category = response.content.strip().lower()
        if "academic" in category:
            category = "academic"
        elif "fee" in category:
            category = "fee"
        else:
            category = "general"
        return {"query_type": category}

    def academic_rag_node(state: State) -> dict:
        query = state["messages"][-1].content
        docs = academic_retriever.invoke(query)
        context = "\n\n".join([doc.page_content for doc in docs])
        return {"retrieved_context": context}

    def fee_rag_node(state: State) -> dict:
        query = state["messages"][-1].content
        docs = fee_retriever.invoke(query)
        context = "\n\n".join([doc.page_content for doc in docs])
        return {"retrieved_context": context}

    def general_node(_: State) -> dict:
        return {"retrieved_context": "NO_RETRIEVAL_NEEDED"}

    def response_node(state: State) -> dict:
        query = state["messages"][-1].content
        programme = state.get("programme", "Unknown")
        context = state["retrieved_context"]

        if context == "NO_RETRIEVAL_NEEDED":
            prompt = (
                f"You are a friendly college assistant talking to a {programme} student. "
                f"Answer this question using your own general knowledge:\n\n{query}"
            )
        else:
            prompt = (
                f"You are a college assistant helping a {programme} student. "
                f"Use the following context from the official college documents to answer "
                f"the question accurately. If the context mentions specific figures for "
                f"different programmes, highlight the one relevant to {programme} if possible.\n\n"
                f"Context:\n{context}\n\n"
                f"Question: {query}\n\n"
                "Give a clear, friendly, and precise answer."
            )

        response = llm.invoke(prompt)
        return {"messages": [("ai", response.content.strip())]}

    def route_query(state: State):
        query_type = state.get("query_type")
        if query_type == "academic":
            return "academic_rag"
        if query_type == "fee":
            return "fee_rag"
        return "general"

    graph = StateGraph(State)
    graph.add_node("classifier", classifier_node)
    graph.add_node("academic_rag", academic_rag_node)
    graph.add_node("fee_rag", fee_rag_node)
    graph.add_node("general", general_node)
    graph.add_node("response", response_node)

    graph.add_edge(START, "classifier")
    graph.add_conditional_edges("classifier", route_query)
    graph.add_edge("academic_rag", "response")
    graph.add_edge("fee_rag", "response")
    graph.add_edge("general", "response")
    graph.add_edge("response", END)
    return graph.compile()


api = FastAPI(title="College Assistant API")

api.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@api.get("/health")
def health_check():
    return {"ok": True}


@api.post("/chat", response_model=ChatResponse)
def chat(payload: ChatRequest):
    if not payload.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    try:
        graph_app = build_graph_app()
        result = graph_app.invoke(
            {
                "programme": payload.programme,
                "messages": [("human", payload.message)],
            }
        )
        answer = result["messages"][-1].content
        return ChatResponse(answer=answer)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


if __name__ == "__main__":
    import uvicorn

    host = os.getenv("API_HOST", "127.0.0.1")
    port = int(os.getenv("API_PORT", "8000"))
    uvicorn.run("api_server:api", host=host, port=port, reload=False)