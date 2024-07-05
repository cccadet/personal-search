import os
from typing import Any, Dict

import streamlit as st
from crewai import Agent, Task
from crewai_tools import tool
from langchain_core.callbacks import BaseCallbackHandler
from langchain_openai import ChatOpenAI
from search.context import utils

llm = ChatOpenAI(model='gpt-3.5-turbo', temperature=0)

# Constantes Globais
AVATARS = {"commentary": "./assets/e-book.png", "bible": "./assets/biblia.png",
           "revisor": "./assets/revisor.png"}
VECTOR_STORE = 'chroma'
PATH_BOOKS = f"./vector-store/{VECTOR_STORE}/books"
PATH_BIBLE = f"./vector-store/{VECTOR_STORE}/bible"
EMBEDDINGS_MODEL = os.getenv("LOCAL_EMBEDDINGS_MODEL")
os.environ["OPENAI_MODEL_NAME"] = 'gpt-3.5-turbo'
DEBUG = os.getenv("DEBUG_MODE", "False").lower() == "true"
VERBOSE = 2 if DEBUG else False

class AgentOutputHandler(BaseCallbackHandler):
    """Custom handler para processar eventos a cada término da chain."""
    def __init__(self, agent_name: str) -> None:
        self.agent_name = agent_name

    def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> None:
        """Retorna a saída da chain, postando uma mensagem com um avatar."""
        st.session_state.messages.append({"role": self.agent_name, "content": outputs['output']})
        st.chat_message(self.agent_name, avatar=AVATARS[self.agent_name]).write(outputs['output'])


# ------- TOOLS ------- #

@tool("Pesquisar em Livros")
def library_tool(text: str) -> str:
    """Retorna livros que possuem conteúdo relacionado com termo ou assunto buscado."""
    result = ""
    db = utils.retriever_context(embeddings_model=EMBEDDINGS_MODEL, path_books=PATH_BOOKS, vector_store=VECTOR_STORE)
    documents = db.similarity_search(text)
    for row in documents:
        page_content = row.page_content
        source = row.metadata['source'].replace('./books/', '').replace('.pdf', '')
        page = row.metadata['page']
        result += f"**Trecho de {source} - página {page}:** \n{page_content}\n\n"
    return result


@tool("Pesquisa na Bíblia")
def bible_tool(text: str) -> str:
    """Retorna trechos bíblicos que possuem conteúdo relacionado com termo ou assunto buscado."""
    result = ""
    db = utils.retriever_bible(embeddings_model=EMBEDDINGS_MODEL, path_books=PATH_BIBLE, vector_store=VECTOR_STORE)
    documents = db.similarity_search(text)
    for row in documents:
        page_content = row.page_content
        source = row.metadata['source'].replace('./bible/', '').replace('.pdf', '')
        page = row.metadata['page']
        result += f"**Trecho de {source} - página {page}:** \n{page_content}\n\n"
    return result


# ------- AGENTS ------- #

library_agent = Agent(
    role="Especialista em Pesquisa em Biblioteca",
    goal="Encontre os trechos de livros que falem explicitamente do tópico fornecido.",
    backstory=(
        "Você atualmente é encarregado de usar a library_tool "
        "para encontrar trechos que falem do tópico fornecido. "
        "Seu objetivo é garantir que os trechos que você fornece "
        "tratem diretamente do tópico e não uma relação distante. "
    ),
    tools=[library_tool],
    allow_delegation=False,
    verbose=VERBOSE,
    llm=llm
)

revisor_agent = Agent(
    role="Revisor de Texto",
    goal="Revisar e melhorar a qualidade dos trechos fornecidos pelos Especialistas.",
    backstory=(
        "Você atualmente é encarregado de revisar e melhorar a qualidade "
        "dos trechos fornecidos pelos Especialistas em Pesquisa da Biblioteca. "
        "Seu objetivo é garantir que os trechos fornecidos são relevantes, "
        "precisos e falem do tópico fornecido pelo cliente."
    ),
    allow_delegation=False,
    verbose=VERBOSE,
    llm=llm,
    callbacks=[AgentOutputHandler("revisor")]
)

commentary_agent = Agent(
    role="Especialista em Comentário Literário",
    goal="Forneça comentários perspicazes sobre os trechos fornecidos.",
    backstory=(
        "Você atualmente é encarregado de fornecer comentários perspicazes "
        "em trechos de livros selecionados pelo Revisor de texto. "
        "Seu objetivo é garantir que seu comentário explique a "
        "relação do trecho do livro com o texto fornecido e ajuda a melhorar "
        "compreensão e apreciação do material."
    ),
    verbose=VERBOSE,
    allow_delegation=False,
    llm=llm,
    callbacks=[AgentOutputHandler("commentary")],
)

bible_agent = Agent(
    role="Especialista em Pesquisa Bíblica",
    goal="Encontre os trechos da Bíblia mais relevantes relacionados ao texto fornecido.",
    backstory=(
        "Você atualmente é encarregado de usar a bible_tool "
        "para encontrar trechos relevantes da Bíblia relacionados "
        "para um texto fornecido. "
        "Seu objetivo é garantir que os trechos que você fornece "
        "são altamente relevantes e úteis para análises posteriores. "
    ),
    tools=[bible_tool],
    allow_delegation=False,
    verbose=VERBOSE,
    llm=llm,
    callbacks=[AgentOutputHandler("bible")],
)

final_revisor_agent = Agent(
    role="Revisor de Texto",
    goal="Revisar e melhorar a qualidade dos trechos fornecidos pelos Especialistas.",
    backstory=(
        "Você atualmente é encarregado de sumarizar os trechos fornecidos pelos especialistas, "
        "deixando claro qual trecho esta sendo avaliado e explicar como o trecho se relaciona com o texto original. "
        "Faça isso em forma de tópicos, para facilitar a compreensão do texto."
    ),
    allow_delegation=False,
    verbose=VERBOSE,
    llm=llm
)



# ------- TASKS ------- #

library_search_task = Task(
    description=(
        "Sua tarefa é usar a library_tool para encontrar trechos de livros que falem explicitamente "
        " do tópico: {text} "
    ),
    expected_output=(
        "Trechos de livros que falem explicitamente do tópico fornecido. "
        "Cada trecho deve ser claramente citado e incluída sua fonte, com o título do livro "
        "autor e número da página."
    ),
    agent=library_agent,
)

library_select_task = Task(
    description=(
        "Selecione os trechos que falam sobre o tópico: {text} "
        "Explique a relação ou indique caso não haja relação. Responda em português."
    ),
    expected_output=(
        "Tópico fornecido pelo cliente."
        "Trechos que falam do tópico fornecido pelo cliente, citando com sua fonte, incluindo o título do livro "
        "autor e número da página ou indicação caso não haja relação. "
        "Explicação onde o trecho fala diretamente sobre o tópico fornecido pelo cliente."
    ),
    agent=revisor_agent,
)

literary_commentary_task = Task(
    description=(
        "Caso haja trechos de livros relacionado, faça um comentário que ajude a melhorar a "
        "compreensão e apreciação do material. Certifique-se de que seu comentário esteja "
        "diretamente relacionado ao texto buscado: {text}"
        "Responda em português."
    ),
    expected_output=(
        "Um comentário detalhado para cada trecho fornecido. "
        "Cada comentário deve deixar claro qual trecho esta sendo avaliado e explicar "
        "como o trecho se relaciona com o texto original "
        "e quais insights ou adicionais contexto que ele fornece. "
    ),
    agent=commentary_agent,
)

bible_search_task = Task(
    description=(
        "O cliente forneceu o seguinte texto para análise:\n"
        "{text}\n\n"
        "Sua tarefa é usar a bible_tool para encontrar os mais relevantes " 
        "trechos da Bíblia relacionados a este texto. "
    ),
    expected_output=(
        "Trechos da Bíblia diretamente relacionados ao texto fornecido. "
        "Cada trecho deve ser claramente citado com sua fonte, incluindo o título do livro "
        ", número da página e se possível capítulo e versículo. Os trechos devem ajudar "
        "a fornecer insights mais profundos ou contexto adicional ao texto fornecido."
    ),
    agent=bible_agent,
)

revisor_task = Task(
    description=(
        "Sua tarefa é revisar e melhorar a qualidade dos trechos fornecidos pelos "
        "Especialista em Comentário Literário e Especialista em Pesquisa Bíblica. "
        "Certifique-se de que os trechos fornecidos são relevantes, precisos e úteis "
        "para análises posteriores. "
        "Responda em português."
    ),
    expected_output=(
        "Faça um resumo dos trechos fornecidos pelos especialistas, deixando claro qual trecho "
        "esta sendo avaliado e explicar como o trecho se relaciona com o texto original. "
        "Faça isso em forma de tópicos, para facilitar a compreensão do texto."
    ),
    agent=final_revisor_agent,
    context=[library_select_task, literary_commentary_task, bible_search_task]
)
