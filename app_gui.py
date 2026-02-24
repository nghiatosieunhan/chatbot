import streamlit as st
import os
from src.chatbot_agentic import OptimizedAgenticRAG 
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = "./vectorstore/chromadb"
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

@st.cache_resource
def load_rag_agent():
    if not GOOGLE_API_KEY:
        st.error("GOOGLE_API_KEY is missing!")
        return None
    return OptimizedAgenticRAG(db_path=DB_PATH, api_key=GOOGLE_API_KEY)

st.title("🤖 Universal PDF Intelligence Agent")

agent = load_rag_agent()

if agent:
    st.sidebar.success("Database Connected!")
    st.divider()
    st.markdown("### About")
    st.info("This system uses Semantic Chunking and Agentic Reranking to ensure high accuracy.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask me anything about your documents..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Assistant is thinking..."):
            try:
                response = agent.ask(prompt)
                full_response = response['answer']                
                st.markdown(full_response)
                
                if response.get('retrieved_contexts'):
                    with st.expander("Show retrieved context"):
                        for i, ctx in enumerate(response['retrieved_contexts']):
                            st.caption(f"Source {i+1}: {ctx[:300]}...")

                st.session_state.messages.append({"role": "assistant", "content": full_response})
                
            except Exception as e:
                st.error(f"An error occurred during processing: {e}")