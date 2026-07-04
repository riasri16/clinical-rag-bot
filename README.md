# clinical-rag-bot — RAG-based Medical Chatbot

A safety-aware medical chatbot built using Retrieval-Augmented Generation (RAG). It answers health-related queries by retrieving relevant context from a knowledge base of medical documents and generating grounded, context-aware responses — reducing hallucination compared to a standalone LLM.

## 🚀 Live Demo
https://huggingface.co/spaces/riasri16/clinical-rag-bot

## Features

- **Retrieval-Augmented Generation (RAG)** — retrieves relevant document chunks before generating an answer, keeping responses grounded in source material
- **Emergency detection** — flags critical symptoms (e.g. chest pain, unconsciousness) and immediately advises the user to seek emergency care instead of querying the LLM
- **Source citation** — every answer shows which document and page it was retrieved from
- **Medical disclaimer** — reminds users the chatbot is not a substitute for professional medical advice
- **Conversational UI** — built with Streamlit, includes chat history, loading states, and a sidebar with a chat reset option

## Tech Stack

- **Python**
- **LangChain** — orchestration framework for the RAG pipeline
- **HuggingFace** — sentence-transformers for embeddings (`all-MiniLM-L6-v2`)
- **FAISS** — vector database for similarity search
- **Groq (LLaMA)** — LLM used for response generation
- **Streamlit** — chatbot UI
- **PyPDF** — PDF document loading and parsing

## Project Architecture

The project is built in three phases:

**Phase 1 — Build the Knowledge Base**
- Load raw PDF documents
- Split documents into chunks
- Generate vector embeddings for each chunk
- Store embeddings in a FAISS vector database

**Phase 2 — Connect Knowledge Base with LLM**
- Load the LLM
- Connect the LLM to the FAISS retriever
- Build a Retrieval-QA chain

**Phase 3 — Chatbot Interface**
- Streamlit-based chat UI
- Load the FAISS vector store (cached)
- Retrieve relevant context and generate an answer using RAG

## Project Structure
medibot-rag/
├── medibot.py                     # Streamlit chatbot application
├── create_memory_for_llm.py       # Builds the FAISS vector store from PDFs
├── connect_memory_with_llm.py     # Standalone script to test the RAG pipeline via CLI
├── requirements.txt               # Python dependencies
└── README.md
> Note: The `data/` (source PDFs) and `vectorstore/` (FAISS index) directories are not included in this repository. Run `create_memory_for_llm.py` locally to generate them from your own documents.

## Setup & Usage

1. Clone the repository
```bash
git clone https://github.com/riasri16/medibot-rag.git
cd medibot-rag
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Add your documents to a `data/` folder and generate the vector store
```bash
python create_memory_for_llm.py
```

4. Set your Groq API key as an environment variable
```bash
export GROQ_API_KEY="your_api_key_here"
```

5. Run the chatbot
```bash
streamlit run medibot.py
```

## Future Improvements

- User authentication
- Self-upload document functionality within the UI
- Multi-document ingestion and merging
- Unit testing for the RAG pipeline
- Multi-language support

## Disclaimer

This chatbot is built for educational purposes and provides general information only. It is not a substitute for professional medical advice, diagnosis, or treatment.
