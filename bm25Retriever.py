from pyserini.search.lucene import LuceneSearcher
import ir_datasets

def generate_bm25_run_file(dataset_name, index_name, output_file, top_k):
    """
    Generate a BM25 run file in TREC format.

    Args:
        dataset_name (str): Name of the dataset to load with ir_datasets.
        index_name (str): Name of the Pyserini prebuilt index to use.
        output_file (str): Path to save the BM25 run file.
        top_k (int): Number of top documents to retrieve for each query.

    Returns:
        None
    """
    # Load the dataset
    dataset = ir_datasets.load(dataset_name)

    # Initialize the BM25 searcher with the prebuilt index
    searcher = LuceneSearcher.from_prebuilt_index(index_name)

    # Generate the BM25 run file
    with open(output_file, 'w') as f:
        for query in dataset.queries_iter():
            query_id = query.query_id
            query_text = query.text

            # Perform BM25 search
            hits = searcher.search(query_text, k=top_k)

            # Write results to TREC format
            for rank, hit in enumerate(hits):
                f.write(f"{query_id} Q0 {hit.docid} {rank + 1} {hit.score} BM25\n")
