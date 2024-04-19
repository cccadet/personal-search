from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
import os
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain.memory import ConversationBufferMemory
from langchain_core.output_parsers import StrOutputParser

embeddings = OllamaEmbeddings(model='nomic-embed-text:v1.5')
faiss_path = "./faiss_index"
db = FAISS.load_local(faiss_path, embeddings, allow_dangerous_deserialization=True)
print(db.index.ntotal)

retriever = db.as_retriever()
retriever.search_kwargs["distance_metric"] = "cos"
retriever.search_kwargs["fetch_k"] = 100
retriever.search_kwargs["maximal_marginal_relevance"] = True
retriever.search_kwargs["k"] = 3

template = '''
    Você é um assistente de busca de livros.

    Histórico de conversa: {chat_history}

    Pergunta: {question}
    '''
prompt = PromptTemplate(
    input_variables=["chat_history", "question"], template=template
)

llm = Ollama(model="mistral:7b")

#memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, input_key="question", output_key='answer', max_memory_length=3)

#todo: memoria de conversa

conversation_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    verbose=True,
    return_source_documents=False,
)

chain = (
    conversation_chain
    | {"question": RunnablePassthrough()}
    | StrOutputParser()
)
