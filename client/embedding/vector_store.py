import os
import shutil
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

#Utilizing the Chroma vector store for embedding and persistence
def initialize_vector_store(split_docs, persist_directory="./client/chroma_db"): 
    return Chroma.from_documents(
        documents=split_docs,
        embedding=embeddings,
        persist_directory=persist_directory
    )


def clear_chroma_db():
    persist_directory = "./chroma_db"
    if os.path.exists(persist_directory):
        try:
            shutil.rmtree(persist_directory)
            print("ChromaDB cleared.")
        except PermissionError:
            print("Fetching fromm current ChromaDb session. Restart server to clear ChromaDB.")
        except KeyError:
            print("ChromaDB cleared.")
