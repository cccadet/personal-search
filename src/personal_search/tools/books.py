"""
This module contains the tools for searching books and videos on Youtube.
"""

from crewai_tools import BaseTool
from embedchain import App

def clean_result(result):
    result_clean = ""
    for r in result:
        source = r['metadata']['source'].replace('./ingestion/pdf/books/','').replace('.pdf','')
        page = str(r['metadata']['page'])
        context = r['context']
        result_clean += source + " - Página " + page + ":\n" + context + "\n\n"
    return result_clean


class LibraryTool(BaseTool):
    name: str = "Pesquisar em Livros"
    description: str = (
        "Retorna livros que possuem conteúdo relacionado com o termo ou assunto buscado."
    )

    def _run(self, topic: str) -> str:
        # load chroma configuration from yaml file
        app = App.from_config(config_path="ingestion/books.yaml")

        result = app.search(query=topic)
        result = clean_result(result)
        return result


def clean_youtube_result(result):
    result_clean = ""
    for r in result:
        source = r['metadata']['title'] + " - " + r['metadata']['author']
        context = r['context']
        result_clean += source + "\n\n" + context + "\n\n"
    return result_clean

class YoutubeTool(BaseTool):
    name: str = "Pesquisa Youtube"
    description: str = (
        "Retorna as transcrições dos vídeos que possuem conteúdo relacionado com termo ou assunto buscado."
    )

    def _run(self, topic: str) -> str:
        app = App.from_config(config_path="ingestion/youtube.yaml")

        result = app.search(query=topic)
        result = clean_youtube_result(result)
        return result
