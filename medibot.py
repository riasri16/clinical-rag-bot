import os
import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint
from langchain_groq import ChatGroq

DB_FAISS_PATH = "vectorstore/db_faiss"

EMERGENCY_KEYWORDS = [
    "chest pain", "severe bleeding", "can't breathe", "cannot breathe",
    "unconscious", "suicide", "heart attack", "stroke", "overdose", "seizure"
]

DISCLAIMER = "⚕️ MediBot provides general information only and is not a substitute for professional medical advice."


def check_emergency(query):
    query_lower = query.lower()
    return any(keyword in query_lower for keyword in EMERGENCY_KEYWORDS)


@st.cache_resource
def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db


def set_custom_prompt(custom_prompt_template):
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
    return prompt


def load_llm(huggingface_repo_id, HF_TOKEN):
    llm = HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        temperature=0.5,
        model_kwargs={"token": HF_TOKEN, "max_length": "512"}
    )
    return llm


def main():
    st.set_page_config(page_title="MediBot", page_icon="⚕️", layout="centered")

    # --- Sidebar ---
    with st.sidebar:
        st.header("⚕️ About MediBot")
        st.write("Ask health-related questions and get answers grounded in verified medical documents.")
        st.divider()
        if st.button("🗑️ Clear Chat"):
            st.session_state.messages = []
            st.rerun()

    st.title("Ask Chatbot!")

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # --- Empty state suggestion ---
    if not st.session_state.messages:
        st.info("👋 Hi! I'm MediBot. Ask me things like:\n- What are the symptoms of diabetes?\n- How is hypertension treated?")

    # --- Render past messages ---
    for message in st.session_state.messages:
        avatar = "🧑" if message['role'] == 'user' else "⚕️"
        st.chat_message(message['role'], avatar=avatar).markdown(message['content'])

    prompt = st.chat_input("Pass your prompt here")

    if prompt:
        st.chat_message('user', avatar="🧑").markdown(prompt)
        st.session_state.messages.append({'role': 'user', 'content': prompt})

        # --- Emergency check before hitting the LLM ---
        if check_emergency(prompt):
            emergency_msg = "⚠️ **This sounds like a medical emergency.** Please call your local emergency number or visit the nearest hospital immediately. This chatbot cannot provide emergency assistance."
            st.chat_message('assistant', avatar="⚕️").markdown(emergency_msg)
            st.session_state.messages.append({'role': 'assistant', 'content': emergency_msg})
            return

        CUSTOM_PROMPT_TEMPLATE = """
                Use the pieces of information provided in the context to answer user's question.
                If you dont know the answer, just say that you dont know, dont try to make up an answer. 
                Dont provide anything out of the given context
                Context: {context}
                Question: {question}
                Start the answer directly. No small talk please.
                """

        with st.chat_message('assistant', avatar="⚕️"):
            with st.spinner("MediBot is thinking..."):
                try:
                    vectorstore = get_vectorstore()
                    if vectorstore is None:
                        st.error("Failed to load the vector store")
                        return

                    qa_chain = RetrievalQA.from_chain_type(
                        llm=ChatGroq(
                            model_name="meta-llama/llama-4-maverick-17b-128e-instruct",
                            temperature=0.0,
                            groq_api_key=os.environ["GROQ_API_KEY"],
                        ),
                        chain_type="stuff",
                        retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
                        return_source_documents=True,
                        chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
                    )
                    response = qa_chain.invoke({'query': prompt})
                    result = response["result"]
                    source_documents = response["source_documents"]

                    st.markdown(result)

                    # --- Sources in expander ---
                    with st.expander("📄 View Sources"):
                        for doc in source_documents:
                            source_name = doc.metadata.get('source', 'Unknown').split('/')[-1]
                            page = doc.metadata.get('page', 'N/A')
                            st.caption(f"{source_name} — Page {page}")

                    st.session_state.messages.append({'role': 'assistant', 'content': result})

                except Exception as e:
                    st.error(f"Error: {str(e)}")

    # --- Footer disclaimer ---
    st.markdown("---")
    st.caption(DISCLAIMER)


if __name__ == "__main__":
    main()