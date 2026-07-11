import os
from dotenv import load_dotenv
from typing import Literal, TypedDict, Annotated
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

academic_retriever = build_retriever("academics_handbook.pdf")
fee_retriever = build_retriever("fee_structure.pdf")

llm = ChatGroq(model = "llama-3.3-70b-versatile", temperature=0.4)

#STEP 2 -> BUILDING STATEE

class State(TypedDict):
    programme : str
    messages : Annotated[list, add_messages]
    query_type : str
    retrieved_context : str


#STEP 3 -> NODES
#this is basic classifier node which classifies the query into academic, fee or general. Based on the classification, the appropriate retriever will be used to fetch relevant context from the respective PDF documents.

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
            category= "academic"
        elif "fee" in category:
            category= "fee"
        else:
            category= "general"
        
        return {"query_type": category}

#second node : academic retriever

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

#this node generates the final answer to the student's query, personalized using the student's programme. It uses the retrieved context from the previous nodes to provide a clear and friendly response.
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

#STEP 4 -> BUILDING ROUTER FUNCTION
#this function routes the query to the appropriate node based on the query type determined by the classifier node. It checks the 'query_type' in the state and returns the corresponding node name.
def route_query(state: State):
    """Routes the query to the appropriate node based on the query type."""
    query_type = state.get("query_type")
    if query_type == "academic":
        return "academic_rag"
    elif query_type == "fee":
        return "fee_rag"
    else:
        return "general"
    
#STEP 5 -> BUILDING THE STATE GRAPH
#this is the main state graph that defines the flow of the conversation. It starts with the

graph = StateGraph(State)
graph.add_node("classifier", classifier_node)
graph.add_node("academic_rag", academic_rag_node)
graph.add_node("fee_rag", fee_rag_node)
graph.add_node("general", general_node)
graph.add_node("response", response_node)

#edges
graph.add_edge(START, "classifier")

graph.add_conditional_edges("classifier", route_query)

graph.add_edge("academic_rag", "response")
graph.add_edge("fee_rag", "response")
graph.add_edge("general", "response")

graph.add_edge("response", END)

app = graph.compile()

#STEP 6 -> RUNNING THE STATE GRAPH
#this function runs the state graph, taking in the student's query and programme as input. It

print("Welcome to the College Assistant! Please enter your query and programme.")
print("Which programme are you enrolled in? ")
print("1. BCA")
print("2. BBA")
print("3. BCom")
choice = input("\nEnter the number corresponding to your programme: ")
programme_map = {"1": "BCA", 
                 "2": "BBA", 
                 "3": "BCom"
                 }
student_programme = programme_map.get(choice, "BCA")  # Default to BCA if invalid choice
print(f"You have selected: {student_programme}\n")
print("=" * 50 )

while True:
    user_query = input("You: ")
    if user_query.lower() in ["exit", "quit"]:
        print("Thank you for using the College Assistant. Goodbye!")
        break
    result = app.invoke({
        "programme": student_programme,
        "messages": [("human", user_query)]
    })
    print(f"Assistant: {result['messages'][-1].content}\n")