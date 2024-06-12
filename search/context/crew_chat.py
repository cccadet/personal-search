import os
from typing import Any, Dict

import streamlit as st
from crewai import Agent, Task
from crewai_tools import tool
from langchain_core.callbacks import BaseCallbackHandler
from search.context import utils

# Constantes Globais
AVATARS = {"commentary": "./assets/e-book.png", "bible": "./assets/biblia.png"}
PATH_BOOKS = "./vector-store/faiss/books"
PATH_BIBLE = "./vector-store/faiss/bible"
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
    db = utils.retriever_context(embeddings_model=EMBEDDINGS_MODEL, path_books=PATH_BOOKS, vector_store='faiss')
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
    db = utils.retriever_context(embeddings_model=EMBEDDINGS_MODEL, path_books=PATH_BIBLE, vector_store='faiss')
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
    goal="Encontre os trechos de livros mais relevantes relacionados ao texto fornecido.",
    backstory=(
        "Você atualmente é encarregado de usar a library_tool "
        "para encontrar trechos relevantes de livros relacionados "
        "para um texto fornecido. "
        "Seu objetivo é garantir que os trechos que você fornece "
        "são altamente relevantes e úteis para análises posteriores. "
    ),
    tools=[library_tool],
    allow_delegation=False,
    verbose=VERBOSE,
)

commentary_agent = Agent(
    role="Especialista em Comentário Literário",
    goal="Forneça comentários perspicazes sobre os trechos fornecidos pelo "
         "Especialista em Pesquisa da Biblioteca.",
    backstory=(
        "Você atualmente é encarregado de fornecer comentários perspicazes "
        "em trechos de livros encontrados pelo Especialista em Pesquisa da Biblioteca. "
        "Seu objetivo é garantir que seu comentário seja direto "
        "relacionado ao texto fornecido e ajuda a melhorar "
        "compreensão e apreciação do material."
    ),
    verbose=VERBOSE,
    allow_delegation=False,
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
    callbacks=[AgentOutputHandler("bible")],
)




# ------- TASKS ------- #

library_search_task = Task(
    description=(
        "O cliente forneceu o seguinte texto para análise:\n"
        "{text}\n\n"
        "Sua tarefa é usar a library_tool para encontrar os mais relevantes " 
        "trechos de livros relacionados a este texto. Certifique-se de que os trechos "
        "que você fornece são altamente relevantes e úteis para análises posteriores."
    ),
    expected_output=(
        "Trechos de livros diretamente relacionados ao texto fornecido. "
        "Cada trecho deve ser claramente citado com sua fonte, incluindo o título do livro "
        "autor e número da página. Os trechos devem ajudar a fornecer insights mais profundos "
        "ou contexto adicional ao texto fornecido."
    ),
    agent=library_agent,
)

literary_commentary_task = Task(
    description=(
        "Sua tarefa é citar os trechos retornados pelo"
        "Especialista em Pesquisa da Biblioteca para a pesquisa {text}"
        "e fornecer comentários perspicazes sobre eles. "
        "Certifique-se de que seu comentário esteja diretamente relacionado ao texto buscado "
        "e ajude a melhorar a compreensão e apreciação do material. "
        "Responda em português."
    ),
    expected_output=(
        "Texto original fornecido pelo cliente, "
        "seguido pelos trechos de livros fornecidos pelo Especialista em "
        "Pesquisa da Biblioteca. Por fim, um comentário detalhado sobre os trechos fornecidos. "
        "Cada comentário deve explicar como o trecho se relaciona com o texto original "
        "e quais insights ou adicionais contexto que ele fornece. "
    ),
    agent=commentary_agent,
)

bible_search_task = Task(
    description=(
        "O cliente forneceu o seguinte texto para análise:\n"
        "{text}\n\n"
        "Sua tarefa é usar a bible_tool para encontrar os mais relevantes " 
        "trechos da Bíblia relacionados a este texto. Certifique-se de que os trechos "
        "que você fornece são altamente relevantes e úteis para análises posteriores."
    ),
    expected_output=(
        "Trechos da Bíblia diretamente relacionados ao texto fornecido. "
        "Cada trecho deve ser claramente citado com sua fonte, incluindo o título do livro "
        ", número da página e se possível capítulo e versículo. Os trechos devem ajudar "
        "a fornecer insights mais profundos ou contexto adicional ao texto fornecido."
    ),
    agent=bible_agent,
)
