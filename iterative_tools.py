import os 
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch
from dotenv import load_dotenv


load_dotenv()

#tools 

search_tool = TavilySearch(max_results = 3)
tools = [search_tool]

#llms 

#writer
writer_llm =  ChatGroq(model = "llama-3.3-70b-versatile", temperature=0.3)

writer_llm_with_tools = writer_llm.bind_tools(tools)

#reviewer

reviewer_llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.1)

#state building 

class State(TypedDict):
    topic : str 
    messages : Annotated[list,add_messages]
    draft : str 
    review_feedback : str
    is_approved : bool 
    attempt : int
