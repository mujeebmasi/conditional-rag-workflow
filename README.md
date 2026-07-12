#  Conditional RAG Workflow using LangGraph

A Retrieval-Augmented Generation (RAG) application built with **LangGraph**, **LangChain**, **FAISS**, and **Groq LLM** that intelligently decides whether a user's query requires document retrieval or can be answered directly by the LLM.

##  Features

-  Loads and processes multiple PDF documents
-  Converts documents into vector embeddings
-  Stores embeddings in a FAISS Vector Database
-  Conditional routing using LangGraph
-  Direct LLM response for general queries
-  Retrieval-Augmented Generation (RAG) for document-related queries
-  Streamlit-based interactive UI
-  Fast inference using Groq LLMs

---

##  Workflow

```text
                User Query
                     │
                     ▼
           LangGraph Router Node
                     │
        ┌────────────┴────────────┐
        │                         │
        ▼                         ▼
 General Question          Document Question
        │                         │
        ▼                         ▼
   Groq LLM                FAISS Retrieval
        │                         │
        ▼                         ▼
      Response          Retrieved Context
                                │
                                ▼
                           Groq LLM
                                │
                                ▼
                            Final Answer
```

---

## 🛠 Tech Stack

- Python
- LangChain
- LangGraph
- FAISS
- HuggingFace Embeddings
- Groq LLM
- Streamlit

---

## 📂 Project Structure

```
.
├── app.py                  # Streamlit application
├── main.py                 # Main LangGraph workflow
├── iterative_tools.py      # Helper functions/tools
├── humanintheloop.py       # Human-in-the-loop workflow
├── academics_handbook.pdf
├── fee_structure.pdf
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

Clone the repository

```bash
git clone https://github.com/mujeebmasi/conditional-rag-workflow.git

cd conditional-rag-workflow
```

Create a virtual environment

```bash
python -m venv .venv
```

Activate it

Windows

```bash
.venv\Scripts\activate
```

Linux / Mac

```bash
source .venv/bin/activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🔑 Environment Variables

Create a `.env` file and add your API key.

```env
GROQ_API_KEY=your_api_key_here
```

---

## ▶️ Run the Application

```bash
streamlit run app.py
```

---

## 📖 Example Queries

### Routed directly to the LLM

- What is LangGraph?
- Explain vector databases.
- What is Retrieval-Augmented Generation?

### Routed to RAG

- What is the fee structure?
- What are the attendance rules?
- Summarize the academics handbook.
- Explain the scholarship policy.

---

##  How Conditional Routing Works?

The workflow first analyzes the incoming query.

- If the query is **general knowledge**, it is answered directly by the LLM.
- If the query requires information from uploaded documents, the workflow retrieves the most relevant chunks from the FAISS vector database before generating the response.

This minimizes unnecessary retrieval while improving response quality for document-based questions.

---

##  Future Improvements

- Support multiple uploaded PDFs
- Conversation memory
- Hybrid Search (BM25 + Vector Search)
- Query rewriting
- Source citations
- Evaluation pipeline
- Multi-agent workflows

---

##  Author

**Abdul Mujeeb**

GitHub: https://github.com/mujeebmasi


---


