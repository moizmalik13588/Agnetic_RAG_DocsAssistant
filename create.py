from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os


DATA_PATH = "data/"
DB_FAISS_PATH = "vectorstore/db_faiss"


def load_pdf_files(data_path):
    # print("Loading PDF files from:", data_path)
    loader = DirectoryLoader(data_path, glob='*.pdf', loader_cls=PyPDFLoader)
    documents = loader.load()
    # print(f" Loaded {len(documents)} documents.")
    return documents


def create_chunks(documents):
    # print(" Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    text_chunks = text_splitter.split_documents(documents)
    # print(f" Created {len(text_chunks)} text chunks.")
    return text_chunks


def get_embedding_model():
    print("Loading embedding model...")
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embedding_model


def build_vector_db(chunks, db_path):
    print("Creating FAISS vectorstore...")
    embedding_model = get_embedding_model()
    db = FAISS.from_documents(chunks, embedding_model)


    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    db.save_local(db_path)
    print(f"FAISS vectorstore saved at: {db_path}")


if __name__ == "__main__":
    docs = load_pdf_files(DATA_PATH)
    chunks = create_chunks(docs)
    build_vector_db(chunks, DB_FAISS_PATH)