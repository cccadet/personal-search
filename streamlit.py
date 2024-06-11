# To install required packages:
# pip install crewai==0.22.5 streamlit==1.32.2

import os
import streamlit as st

from crewai import Crew, Process
from langchain_core.callbacks import BaseCallbackHandler
from typing import TYPE_CHECKING, Any, Dict, Optional

from crewai import Crew
from search.context.crew_chat import library_agent, commentary_agent, \
                                library_search_task, literary_commentary_task, \
                                bible_agent, bible_search_task

avators = {"Writer":"https://cdn-icons-png.flaticon.com/512/320/320336.png",
            "Reviewer":"https://cdn-icons-png.freepik.com/512/9408/9408201.png"}

# ------ Ollama - Local---------#
os.environ["LOCAL_EMBEDDINGS"]="http://localhost:11434/api/embeddings"
embeddings_model = "nomic-embed-text:v1.5"
os.environ["LOCAL_EMBEDDINGS_MODEL"]=embeddings_model # documentar como definir variáveis de ambiente

# ------ OpenAI - API ---------#
os.environ["OPENAI_MODEL_NAME"] = 'gpt-3.5-turbo'

# ------ DEBUG ---------#
os.environ["DEBUG"]="False"
debug = os.getenv("DEBUG_MODE", "False")
if debug.lower() == "true":
    VERBOSE = 2
else:
    VERBOSE = False


# ------- AGENTS ------- #


st.title("💬 CrewAI Writing Studio")

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "Qual assunto deseja pesquisar?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input():

    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    # Establishing the crew with a hierarchical process
    # process=Process.hierarchical  # Specifies the hierarchical management approach
    crew = Crew(
        agents=[library_agent, commentary_agent, bible_agent],
        tasks=[library_search_task, literary_commentary_task, bible_search_task],
        verbose=VERBOSE,
        memory=False,
    )
    inputs = {
        "text": prompt
    }
    final = crew.kickoff(inputs=inputs)

    result = f"## Here is the Final Result \n\n {final}"
    st.session_state.messages.append({"role": "assistant", "content": result})
    st.chat_message("assistant").write(result)