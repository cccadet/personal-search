import os
import streamlit as st
from crewai import Crew
from search.context.crew_chat import (
    library_agent, commentary_agent, library_search_task,
    literary_commentary_task, bible_agent, bible_search_task,
    revisor_agent, revisor_task,
    library_select_task, final_revisor_agent
)

# Configurações locais
LOCAL_EMBEDDINGS_URL = "http://localhost:11434/api/embeddings"
EMBEDDINGS_MODEL = "nomic-embed-text:v1.5"
os.environ["LOCAL_EMBEDDINGS"] = LOCAL_EMBEDDINGS_URL
os.environ["LOCAL_EMBEDDINGS_MODEL"] = EMBEDDINGS_MODEL

# Configurações da API OpenAI
os.environ["OPENAI_MODEL_NAME"] = 'gpt-3.5-turbo'

# Configurações de DEBUG
os.environ["DEBUG"] = "False"
DEBUG_MODE = os.getenv("DEBUG_MODE", "False").lower() == "true"
VERBOSE = 2 if DEBUG_MODE else False

# Título da aplicação Streamlit
st.title("💬 Pesquisa Pessoal")

# Inicialização do estado da sessão para mensagens
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant",
                                     "content": "Qual assunto deseja pesquisar? Você tem acesso a sua biblioteca pessoal e a Bíblia."}]

# Exibição das mensagens
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# Entrada do usuário
if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    # Configuração e execução do Crew
    crew = Crew(
        agents=[library_agent, revisor_agent, commentary_agent, bible_agent, final_revisor_agent],
        tasks=[library_search_task, library_select_task, literary_commentary_task, bible_search_task, revisor_task],
        verbose=VERBOSE,
        memory=False,
    )
    inputs = {"text": prompt}
    final = crew.kickoff(inputs=inputs)

    # Exibição do resultado
    result = f"## Aqui está o resultado final \n\n {final}"
    st.session_state.messages.append({"role": "assistant", "content": result})
    st.chat_message("assistant").write(result)