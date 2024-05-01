from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain_core.output_parsers import StrOutputParser
from operator import itemgetter
from langchain.load import dumps, loads

def retriver_context(embeddings_model='nomic-embed-text:v1.5',
                     path_books='./vector-store/books',
                     distance_metric='cos',
                     fetch_k=100,
                     k=3,
                     maximal_marginal_relevance=False):
    """
    Retrieves the context using the specified parameters.

    Args:
        embeddings_model (str): The embeddings model to use. Default is 'nomic-embed-text:v1.5'.
        path_books (str): The path to the vector store. Default is './vector-store/books'.
        distance_metric (str): The distance metric to use for retrieval. Default is 'cos'.
        fetch_k (int): The number of documents to fetch before filtering. Default is 100.
        k (int): The number of documents to return. Default is 3.
        maximal_marginal_relevance (bool): Whether to use maximal marginal relevance for example selection. Default is False.

    Returns:
        tuple: A tuple containing the retriever and the vector store.

    """
    # Load the embeddings model
    embeddings = OllamaEmbeddings(model=embeddings_model)
    # Load the Vector Store
    db = FAISS.load_local(path_books, embeddings, allow_dangerous_deserialization=True)

    # Load vectore store as retriever (search engine)
    retriever = db.as_retriever()
    # distance_metric: cosine
    retriever.search_kwargs["distance_metric"] = distance_metric
    # fetch_k: search method to set how many documents you want to fetch before filtering
    retriever.search_kwargs["fetch_k"] = fetch_k
    # maximal_marginal_relevance: The MaxMarginalRelevanceExampleSelector selects 
    # examples based on a combination of which examples are most similar to the inputs,
    # while also optimizing for diversity.
    retriever.search_kwargs["maximal_marginal_relevance"] = maximal_marginal_relevance
    # k: Number of documents to return
    retriever.search_kwargs["k"] = k
    return retriever, db


def search_context(retriever, prompt, question, model='llama3:8b', temperature=0):
    """
    Searches for context using a retriever and generates an answer to a given question using a language model.

    Args:
        retriever (str): The context retriever.
        prompt (str): The prompt for the language model.
        question (str): The question to be answered.
        model (str, optional): The language model to use. Defaults to 'llama3:8b'.
        temperature (int, optional): The temperature for language model sampling. Defaults to 0.

    Returns:
        str: The generated answer to the question.
    """
    llm = Ollama(model=model, temperature=temperature)

    final_rag_chain = (
        {"context": retriever,
        "question": itemgetter("question")} 
        | prompt
        | llm.bind(stop=['<|eot_id|>']) # stop when the model outputs the end of text token (llama3)
        | StrOutputParser()
    )

    final_answer = final_rag_chain.invoke({"question":question})
    return final_answer


def generate_multi_query(prompt, model='llama3:8b'):
    """
    Generates multiple queries based on a given prompt using the Ollama model.

    Args:
        prompt (str): The prompt for generating queries.
        model (str, optional): The Ollama model to use for query generation. 
        Defaults to 'llama3:8b'.

    Returns:
        list: A list of generated queries.

    """
    llm = Ollama(model=model)

    generate_queries = (
        prompt
        | llm.bind(stop=['<|eot_id|>'])
        | StrOutputParser()
        | (lambda x: x.split("\n"))
    )

    return generate_queries


def get_unique_union(documents: list[list]):
    """
    Returns a list of unique documents by flattening the input list of lists,
    converting each document to a string, and removing duplicates.

    Args:
        documents (list[list]): A list of lists containing documents.

    Returns:
        list: A list of unique documents.

    """
    # Flatten list of lists, and convert each Document to string
    flattened_docs = [dumps(doc) for sublist in documents for doc in sublist]
    # Get unique documents
    unique_docs = list(set(flattened_docs))
    # Return
    return [loads(doc) for doc in unique_docs]