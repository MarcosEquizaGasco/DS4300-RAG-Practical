from pymilvus import connections, utility, Collection, CollectionSchema, FieldSchema, DataType
import os
import string
import re
from src.ingest_redis import get_embedding, extract_text_from_pdf, split_text_into_chunks
import pymilvus
import ast

# Connect to Milvus server
pymilvus.connections.connect(alias="default", host="localhost", port=19530)
def clear_milvus_collection(collection_name="hnsw_index"):
    if utility.has_collection(collection_name):
        utility.drop_collection(collection_name)

class MilvusEmbedding:
    def __init__(self, model_name="nomic-embed-text"):
        self.model_name = model_name

    def __call__(self, input):
        return [get_embedding(text, model=self.model_name) for text in input]

def create_milvus_collection(collection_name="hnsw_index", embed_dim=768, embed_model="nomic-embed-text"):
    clear_milvus_collection(collection_name)

    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=embed_dim)
    ]
    schema = CollectionSchema(fields, description="HNSW Index Collection")
    collection = Collection(name=collection_name, schema=schema)
    collection.create_index(field_name="embedding", index_params={"index_type": "HNSW", "metric_type": "COSINE", "params": {"M": 16, "efConstruction": 200}})
    return collection

def process_pdfs_milvus(collection, data_dir, chunk_size=300, overlap=50, clean=False):
    data = []
    for file_name in os.listdir(data_dir):
        if file_name.endswith(".pdf"):
            pdf_path = os.path.join(data_dir, file_name)
            text_by_page = extract_text_from_pdf(pdf_path)
            for _, text in text_by_page:
                if clean:
                    text = text.translate(str.maketrans('', '', string.punctuation))
                    text = re.sub(' +', ' ', text)

                chunks = split_text_into_chunks(text, chunk_size=chunk_size, overlap=overlap)
                embeddings = [get_embedding(chunk) for chunk in chunks]
                data.extend([(chunk, embedding) for chunk, embedding in zip(chunks, embeddings)])

    if data:
        collection.insert([[item[0] for item in data], [item[1] for item in data]])
        collection.flush()
        collection.load()  
    
    return collection, len(data)

def query_milvus(collection, query_text, top_k=5):
    
    collection.load()

    embedding = get_embedding(query_text)
    search_params = {"metric_type": "COSINE", "params": {"ef": 100}}

    # Perform the search
    results = collection.search([embedding], "embedding", search_params, top_k=top_k, output_fields=["text"], limit=10)

    result_list = []
    for batch in results:
        raw_data = batch[0].entity.get("text")        
        result_list.append(raw_data)

    return result_list
