import os
import streamlit as st
from dotenv import load_dotenv
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()


def configure_groq_api_key() -> None:
    """Load Groq API key from Streamlit secrets first, then environment."""
    groq_key = None
    try:
        groq_key = st.secrets.get("GROQ_API_KEY")
    except Exception:
        # No secrets.toml configured locally; fall back to environment/.env.
        groq_key = None

    if not groq_key:
        groq_key = os.getenv("GROQ_API_KEY")

    if not groq_key:
        raise RuntimeError(
            "Missing GROQ_API_KEY. Add it in Streamlit Cloud app secrets "
            "or set it in a local .env file."
        )
    os.environ["GROQ_API_KEY"] = groq_key


try:
    configure_groq_api_key()
except Exception as key_error:
    st.error(str(key_error))
    st.stop()


# ============================================================
# ALL LOGIC BELOW (retrievers, state, nodes, graph) IS EXACTLY
# THE SAME AS YOUR ORIGINAL SCRIPT. NOTHING HAS BEEN CHANGED
# EXCEPT WRAPPING THE SLOW SETUP IN @st.cache_resource SO IT
# ONLY RUNS ONCE, AND REPLACING THE CLI while-loop/input() WITH
# A STREAMLIT CHAT UI AT THE BOTTOM OF THIS FILE.
# ============================================================

@st.cache_resource(show_spinner="Loading embeddings model...")
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


@st.cache_resource(show_spinner="Building retriever from PDF...")
def build_retriever(pdf_path: str):
    embeddings = get_embeddings()
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_documents(documents)
    vectorstore = FAISS.from_documents(chunks, embeddings)

    return vectorstore.as_retriever(search_kwargs={"k": 4})


class State(TypedDict):
    programme: str
    messages: Annotated[list, add_messages]
    query_type: str
    retrieved_context: str


@st.cache_resource(show_spinner="Setting up the assistant graph...")
def build_app():
    academic_retriever = build_retriever("academics_handbook.pdf")
    fee_retriever = build_retriever("fee_structure.pdf")
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.4)

    def classifier_node(state: State) -> dict:
        """Look at the latest human message and decide which retriever to use based on the query type."""
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
        """Retrieves relevant chunks from the academics handbook."""
        query = state["messages"][-1].content
        docs = academic_retriever.invoke(query)
        context = "\n\n".join([doc.page_content for doc in docs])
        return {"retrieved_context": context}

    def fee_rag_node(state: State) -> dict:
        """Retrieves relevant chunks from the fee structure PDF."""
        query = state["messages"][-1].content
        docs = fee_retriever.invoke(query)
        context = "\n\n".join([doc.page_content for doc in docs])
        return {"retrieved_context": context}

    def general_node(state: State) -> dict:
        """Answers directly using the LLM's own knowledge, no retrieval needed."""
        return {"retrieved_context": "NO_RETRIEVAL_NEEDED"}

    def response_node(state: State) -> dict:
        """Generates the final answer, personalized using the student's programme."""
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
                f"Give a clear, friendly, and precise answer."
            )

        response = llm.invoke(prompt)
        return {"messages": [("ai", response.content.strip())]}

    def route_query(state: State):
        """Routes the query to the appropriate node based on the query type."""
        query_type = state.get("query_type")
        if query_type == "academic":
            return "academic_rag"
        elif query_type == "fee":
            return "fee_rag"
        else:
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


# ============================================================
# STREAMLIT UI
# ============================================================

st.set_page_config(
    page_title="College Assistant",
    page_icon="🎓",
    layout="centered",
)

st.markdown(
    """
    <style>
        .main .block-container {
            max-width: 780px;
            padding-top: 2rem;
        }
        [data-testid="stChatMessage"] {
            border-radius: 14px;
        }
        .header-card {
            background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
            padding: 1.6rem 1.8rem;
            border-radius: 16px;
            color: white;
            margin-bottom: 1.5rem;
        }
        .header-card h1 {
            font-size: 1.6rem;
            margin: 0 0 0.2rem 0;
        }
        .header-card p {
            margin: 0;
            opacity: 0.9;
            font-size: 0.95rem;
        }
        .badge {
            display: inline-block;
            background: rgba(255,255,255,0.18);
            padding: 0.15rem 0.7rem;
            border-radius: 999px;
            font-size: 0.8rem;
            margin-top: 0.5rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---- Sidebar: programme selection ----
with st.sidebar:
    st.markdown("### 🎓 Student Details")
    programme = st.selectbox(
        "Which programme are you enrolled in?",
        options=["BCA", "BBA", "BCom"],
        index=0,
    )
    st.divider()
    st.markdown(
        "This assistant answers questions about **academics** "
        "(attendance, exams, credits, promotion, etc.) and **fees** "
        "(tuition, refunds, scholarships, etc.), routing each question "
        "to the right document automatically."
    )
    st.divider()
    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ---- Header ----
st.markdown(
    f"""
    <div class="header-card">
        <h1>🎓 College Assistant</h1>
        <p>Ask me anything about academics or fees.</p>
        <span class="badge">Programme: {programme}</span>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---- Build graph (cached, runs once) ----
try:
    app = build_app()
except Exception as e:
    st.error(f"Failed to initialize the assistant: {e}")
    st.stop()

# ---- Chat history state ----
if "messages" not in st.session_state:
    st.session_state.messages = []

# ---- Render existing chat history ----
for msg in st.session_state.messages:
    avatar = "🧑‍🎓" if msg["role"] == "user" else "🎓"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])

# ---- Chat input ----
user_query = st.chat_input("Type your question here...")

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user", avatar="🧑‍🎓"):
        st.markdown(user_query)

    with st.chat_message("assistant", avatar="🎓"):
        with st.spinner("Thinking..."):
            try:
                result = app.invoke({
                    "programme": programme,
                    "messages": [("human", user_query)],
                })
                answer = result["messages"][-1].content
            except Exception as chat_error:
                answer = (
                    "I could not get a response from Groq right now. "
                    "Please check your API key, model access, and rate limits, then try again.\n\n"
                    f"Technical details: {chat_error}"
                )
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})