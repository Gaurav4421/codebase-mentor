"""
Streamlit front-end. All actual logic lives in pipeline.CodebaseMentor -- this
file is purely UI wiring + session-state management, so it stays thin and the
same pipeline can be driven from evaluation/benchmark.py or any other caller.

Run with: streamlit run app/streamlit_app.py
"""
import logging
import os
import sys
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

sys.path.append(str(Path(__file__).resolve().parent.parent))  # allow `import pipeline` etc. when run directly

load_dotenv()

from llm.models import GeminiClient
from pipeline import CodebaseMentor

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))

st.set_page_config(page_title="Codebase Mentor", page_icon="🧭", layout="wide")
st.title("🧭 Codebase Mentor")


@st.cache_resource
def get_mentor() -> CodebaseMentor:
    return CodebaseMentor(llm=GeminiClient())


mentor = get_mentor()

if "memory" not in st.session_state:
    st.session_state.memory = None
if "chat" not in st.session_state:
    st.session_state.chat = []  # list of (role, text) for rendering
if "indexed_repo" not in st.session_state:
    st.session_state.indexed_repo = None

with st.sidebar:
    st.header("Repository")
    repo_path = st.text_input("Local path to repository", value=st.session_state.indexed_repo or "")
    if st.button("Index repository", disabled=not repo_path):
        with st.spinner(f"Indexing {repo_path} ..."):
            try:
                repo_index = mentor.index_repository(repo_path)
                st.session_state.indexed_repo = repo_path
                st.session_state.memory = mentor.new_conversation()
                st.session_state.chat = []
                st.success(f"Indexed {len(repo_index.chunks)} chunks.")
            except Exception as e:
                st.error(f"Indexing failed: {e}")

    if st.session_state.indexed_repo:
        st.markdown("**Repository summary**")
        st.write(mentor.index.intelligence.repo_summary)
        with st.expander("Modules"):
            for name, mod in mentor.index.intelligence.modules.items():
                st.markdown(f"- **{name}**: {mod.summary}")

    if st.button("Reset conversation", disabled=not st.session_state.indexed_repo):
        st.session_state.memory = mentor.new_conversation()
        st.session_state.chat = []

for role, text in st.session_state.chat:
    with st.chat_message(role):
        st.markdown(text)

query = st.chat_input(
    "Ask about the codebase..." if st.session_state.indexed_repo else "Index a repository first"
)

if query:
    if not st.session_state.indexed_repo:
        st.warning("Index a repository in the sidebar before asking questions.")
    else:
        st.session_state.chat.append(("user", query))
        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer, intent, _results = mentor.ask(query, st.session_state.memory)
            st.caption(f"intent: {intent}")
            st.markdown(answer)

        st.session_state.chat.append(("assistant", answer))
        st.session_state.memory.add_user_turn(query)
        st.session_state.memory.add_assistant_turn(answer)
        st.session_state.memory.trim()
