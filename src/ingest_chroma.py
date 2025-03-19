import chromadb
from chromadb.api.types import EmbeddingFunction
import os
import string
import re
from src.ingest_redis import get_embedding, extract_text_from_pdf, split_text_into_chunks


chroma_client = chromadb.HttpClient(host='localhost', port=8002)


def clear_chroma_index():

    chroma_client.delete_collection("hnsw_index")

class ChromaEmbedding(EmbeddingFunction):
    def __init__(self, model_name = "nomic-embed-text"):
        self.model_name = model_name

    def __call__(self, input):
        return [get_embedding(text, model=self.model_name) for text in input]



def create_chroma_index(embed_model = "nomic-embed-text"):

    embedding_function = ChromaEmbedding(embed_model)

    try:
        clear_chroma_index()

    except:
        pass

    collection = chroma_client.create_collection(
        name = "hnsw_index",
        embedding_function = embedding_function)
    
    return collection
    
def process_pdfs_chroma(collection, data_dir, chunk_size=300, overlap=50, clean = False):

    chunk_count = 0
    for file_name in os.listdir(data_dir):
        if file_name.endswith(".pdf"):
            pdf_path = os.path.join(data_dir, file_name)
            text_by_page = extract_text_from_pdf(pdf_path)
            for page_num, text in text_by_page:
                if clean:
                    text = text.translate(str.maketrans('', '', string.punctuation))
                    text = re.sub(' +', ' ', text)

                chunks = split_text_into_chunks(text, chunk_size=chunk_size, overlap=overlap)
                ids = [str(i) for i in list(range(chunk_count, chunk_count+len(chunks)))]
                chunk_count += len(chunks)

                collection.add(ids = ids, documents= chunks)

    return collection, chunk_count


def query_chroma(collection, query_text):

    if isinstance(query_text, list):
        result = collection.query(query_texts=query_text,
                        n_results = 5)
    else:
         result = collection.query(query_texts=[query_text],
                        n_results = 5)       
    
    return result