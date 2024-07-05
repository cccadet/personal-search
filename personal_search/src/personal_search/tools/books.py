from crewai_tools import BaseTool
from personal_search.tools import utils
from crewai_tools import YoutubeChannelSearchTool, YoutubeVideoSearchTool

VECTOR_STORE = "chroma"
PATH_BOOKS = f"./vector-store/{VECTOR_STORE}/books"
PATH_BIBLE = f"./vector-store/{VECTOR_STORE}/bible"

class LibraryTool(BaseTool):
    name: str = "Pesquisar em Livros"
    description: str = (
        "Retorna livros que possuem conteúdo relacionado com o termo ou assunto buscado."
    )

    def _run(self, topic: str) -> str:
        result = ""
        db = utils.retriever_context(path_books=PATH_BOOKS, vector_store=VECTOR_STORE)
        documents = db.similarity_search(topic)
        for row in documents:
            page_content = row.page_content
            source = row.metadata['source'].replace('./books/', '').replace('.pdf', '')
            page = row.metadata['page']
            result += f"**Trecho de {source} - página {page}:** \n{page_content}\n\n"
        return result


class BibleTool(BaseTool):
    name: str = "Pesquisa na Bíblia"
    description: str = (
        "Retorna trechos bíblicos que possuem conteúdo relacionado com termo ou assunto buscado."
    )

    def _run(self, topic: str) -> str:
        result = ""
        db = utils.retriever_bible(path_books=PATH_BIBLE, vector_store=VECTOR_STORE)
        documents = db.similarity_search(topic)
        for row in documents:
            page_content = row.page_content
            source = row.metadata['source'].replace('./bible/', '').replace('.pdf', '')
            page = row.metadata['page']
            result += f"**Trecho de {source} - página {page}:** \n{page_content}\n\n"
        return result
    
class DoisDedosTeoTool(BaseTool):
    name: str = "Pesquisa Youtube Dois Dedos de Teologia"
    description: str = (
        "Retorna dados de vídeos do canal Dois Dedos de Teologia que possuem conteúdo relacionado com termo ou assunto buscado."
    )

    def _run(self, topic: str) -> str:
        #tool = YoutubeChannelSearchTool(youtube_channel_handle='@doisdedosdeteologia')
        #result = tool.run(search_query=topic)
        tool_youtube = YoutubeVideoSearchTool(youtube_video_url='https://www.youtube.com/watch?v=BkQdqj-yMj0', config)
        result = tool_youtube.run(search_query=topic)
        return result
