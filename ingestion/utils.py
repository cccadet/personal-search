import os
import logging
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# sset logging level to INFO
logging.basicConfig(level=logging.INFO)

class BookIngestor:
    def __init__(self,
                 library,
                 path_vector_store='./vector_store',
                 chunk_size=500,
                 embeddings=OllamaEmbeddings(model='nomic-embed-text:v1.5')):
        self.library = library
        self.embeddings = embeddings
        self.chunk_size = chunk_size
        self.path_vector_store = path_vector_store

    def ingest(self, books):
        if self.library == 'faiss':
            self.ingest_with_faiss(books)
        elif self.library == 'chroma':
            self.ingest_with_chroma(books)
        else:
            raise ValueError(f'Library {self.library} not supported')

    def ingest_with_chroma(self, books):
        raise NotImplementedError('Chroma library not implemented yet')

    def load_books(self, path_books):
        """
        Load books from the specified path.

        Args:
            path_books (str): The path to the directory containing the books.

        Returns:
            list: A list of file paths for the books found in the directory.
        """
        logging.info('Loading books from %s', path_books)
        books = []
        for root, dirs, files in os.walk(path_books):
            for file in files:
                if file.endswith('.pdf'):
                    books.append(os.path.join(root, file))
        return books

    def load_pdf(self, file):
        """
        Loads a PDF file and returns the extracted documents with PyPDFLoader.

        Args:
            file (str): The path to the PDF file.

        Returns:
            list: A list of extracted documents from the PDF.

        """
        logging.info('Loading PDF %s', file)
        loader = PyPDFLoader(file)
        documents = loader.load()
        return documents

    def load_chunked_docs(self, documents):
        """
        Chunk documents.

        Args:
            documents (list): A list of documents to be chunked.

        Returns:
            list: A list of chunked documents.

        """
        logging.info('Chunking documents')
        text_spliter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=100
        )
        chunked_docs = text_spliter.split_documents(documents)
        return chunked_docs

    def load_vector_store(self, chunked_docs, path_vector_store):
        """
        Loads or creates a vector store using FAISS.

        Args:
            chunked_docs (list): A list of chunked documents.
            path_vector_store (str): The path to the vector store.

        Returns:
            FAISS object: The loaded or created vector store.

        Raises:
            None

        """
        if not os.path.isdir(path_vector_store):
            logging.info('Creating vector store')
            os.makedirs(path_vector_store)
            db = FAISS.from_documents(chunked_docs, self.embeddings)
            db.save_local(path_vector_store)
            logging.info('Vector store loaded %s', db.index.ntotal)
            return db
        else:
            logging.info('Loading vector store')
            db = FAISS.load_local(folder_path=path_vector_store,
                                embeddings=self.embeddings,
                                allow_dangerous_deserialization=True)
            logging.info('Adding documents to vector store')
            db.add_documents(chunked_docs)
            db.save_local(path_vector_store)
            logging.info('Vector store loaded %s', db.index.ntotal)
            return db

    def ingest_with_faiss(self, books):
        # Done path
        path_done = f'{books}/done'
        # Get the last directory of the books
        books_path = books.split('/')[-1]
        # Load books from path
        books = self.load_books(books)
        # Process each book
        for book in books:
            logging.info('Processing book %s', book)
            documents_books = self.load_pdf(book)
            chunked_docs = self.load_chunked_docs(documents_books)
            self.load_vector_store(chunked_docs=chunked_docs,
                path_vector_store=f'{self.path_vector_store}/faiss/{books_path}')
            logging.info('Moving book %s to %s', book, path_done)
            os.rename(book, os.path.join(path_done, os.path.basename(book)))
