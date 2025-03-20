from src.ingest_redis import *
from src.ingest_chroma import *
from src.ingest_milvus import * 
from src.search import *
import time
import pandas as pd


def main():

    # set up dataframe to track experiment results
    cols =  ['database', 'chunk_size', 'overlap', 'clean', 'embedding', 'chunks_processed', 'time_to_process', 'used_memory_mb', 'query_time']
    results = pd.DataFrame(columns = cols)

    with open('example_queries.txt', 'r') as file:

        # Skip lines that don't contain actual queries (headers, empty lines) and extract example queries
        queries = [line.strip() for line in file if line.strip() and not line.strip().startswith('#') and not line.strip().startswith('##')]
        queries = [q.split('. ', 1)[1] if '. ' in q else q for q in queries]
        
        # Print total count
        print(f"Total queries: {len(queries)}")

    # test different chunk/overlap/clean combos for filling redis database
    embedding_model = 'nomic-embed-text'
    embedding_size = 768
    db = 'milvus'

    # loop through different options
    for chunk in [100,300,500,1000]:
        for overlap in [0, 50, 100]:
            if overlap >= chunk:
                continue
            for clean in [True, False]:

                # create and fill Milvus collection
                start = time.time()
                collection = create_milvus_collection(embed_dim=embedding_size)
                collection, num_vectors = process_pdfs_milvus(collection, "data/", chunk_size=chunk, overlap=overlap, clean=clean)
                index_time = time.time() - start
                print(f"Index with chunk size {chunk} and overlap {overlap} created in {round(index_time, 2)} seconds")

                # get memory usage
                float_size = np.dtype(np.float32).itemsize  # 4 bytes per float
                memory_usage_bytes = num_vectors * embedding_size * float_size
                memory_usage_mb = memory_usage_bytes / (1024 * 1024)
                print(f"It uses {round(memory_usage_mb, 2)} MB")

                # test searcging speed
                start = time.time()
                for query in queries:
                    print(f"Collection exists: {utility.has_collection(collection.name)}")
                    print(f"Collection row count: {collection.num_entities}")
                    print(f"Collection loading progress: {utility.loading_progress(collection.name)}")


                    query_milvus(collection, query, top_k=5)
                search_time = time.time() - start
                print(f"Search with chunk size {chunk} and overlap {overlap} completed in {round(search_time, 2)} seconds")

                # add results to resul dataframe
                new_row = [db, chunk, overlap, clean, embedding_model, num_vectors, index_time, memory_usage_mb, search_time]
                results = pd.concat([results, pd.DataFrame([new_row], columns=cols)], ignore_index=True)


if __name__ == "__main__":
    main()