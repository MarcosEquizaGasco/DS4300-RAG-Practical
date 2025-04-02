from src.ingest_redis import *
from src.ingest_chroma import *
from src.ingest_milvus import * 
from src.search import *
import time
import pandas as pd


def main():
    # set up variables for pipeline
    embedding_model = 'nomic-embed-text' # can also choose: 'mxbai-embed-large', 'bge--m3
    llm_model = 'llama3.2:latest' # can also choose mistral:latest, deepseek-r1:7b
    vector_db = "chroma" # can also choose "milvus", "redis"
    chunk_size = 500 # recommended chunk size is 500, will work with 100-1000
    overlap = 100 # recommended is 100 for chunk_size 500, will work with 50-200
    clean = True

    # create collection
    if vector_db == "chroma":
        collection = create_chroma_index(embedding_model)
        collection, chunk_count = process_pdfs_chroma(collection, "data/", chunk_size=chunk_size, overlap=overlap, clean = clean)
    elif vector_db == "milvus":
        collection = create_milvus_collection(embed_model=embedding_model)
        collection, chunk_count = process_pdfs_milvus(collection, "data/", chunk_size=chunk_size, overlap=overlap, clean = clean)
    elif vector_db == "redis":
        collection = create_hnsw_index()
        chunk_count = process_pdfs_redis("data/", chunk_size=chunk_size, overlap=overlap, clean = clean, model = embedding_model)

    # run search
    interactive_search(vector_db, embed_model=embedding_model, llm=llm_model)


if __name__ == "__main__":
    main()