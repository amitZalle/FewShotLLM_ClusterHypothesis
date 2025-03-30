from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pointwise import PointwiseLlmRanker
from rankers import SearchResult
import json


def parse_run_file_with_search_results(run_path, ret_list_size, docstore):
    """
    Parses a TREC-format run file and extracts the top-k passages per query as SearchResult objects.

    Args:
        run_path (str): Path to the TREC-formatted run file containing ranked lists.
        ret_list_size (int): The maximum number of passages to retrieve per query.
        docstore (ir_datasets.docs_store): An ir_datasets document store used to retrieve passage texts.

    Returns:
        dict: A dictionary where each key is a query ID (str) and the corresponding value
              is a list of SearchResult objects containing document IDs, texts, and scores.
    """
    query_passages = {}
    with open(run_path, 'r') as f:
        for line in f:
            query_id, _, doc_id, rank, score, _ = line.strip().split()
            rank = int(rank)
            score = float(score)

            if rank <= ret_list_size:
                doc_text = docstore.get(doc_id).text if docstore else f"Text for doc_id {doc_id}"

                if query_id not in query_passages:
                    query_passages[query_id] = []

                search_result = SearchResult(docid=doc_id, text=doc_text, score=score)
                query_passages[query_id].append(search_result)
    return query_passages


def retrieve_tfidf_nn(passages, n):
    """
    Retrieve the top-n nearest neighbors for each passage using TF-IDF.

    Args:
        passages (list): List of SearchResult objects for the query.
        n (int): Number of nearest neighbors to retrieve.

    Returns:
        dict: A dictionary where keys are doc_ids and values are lists of SearchResult objects (NN clusters).
    """
    texts = [p.text for p in passages]

    # Compute TF-IDF vectors
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(texts)

    # Create the NN clusters
    nn_clusters = {}
    for i, passage in enumerate(passages):
        similarities = cosine_similarity(tfidf_matrix[i], tfidf_matrix).flatten()
        top_n_indices = similarities.argsort()[-(n + 1):-1][::-1]
        nn_cluster = [passages[idx] for idx in top_n_indices]
        nn_clusters[passage.docid] = nn_cluster

    return nn_clusters


def retrieve_llm_nn(passages, ranker, n):
    """
    Retrieve the top-n nearest neighbors for each passage using an LLM-based ranker.

    Args:
        passages (list): List of SearchResult objects for the query.
        ranker (PointwiseLlmRanker): Fine-tuned LLM ranker for scoring passages.
        n (int): Number of nearest neighbors to retrieve.

    Returns:
        dict: A dictionary where keys are doc_ids and values are lists of SearchResult objects (NN clusters).
    """
    nn_clusters = {}

    for query_passage in passages:
        query_text = query_passage.text

        candidates = [p for p in passages if p.docid != query_passage.docid]  # Exclude the query passage itself
        reranked_candidates = ranker.rerank(query_text, candidates)

        nn_clusters[query_passage.docid] = reranked_candidates[:n]

    return nn_clusters


def load_qrels(qrels_path, relevance_threshold=2):
    """
    Load qrels from a file and parse into a dictionary.

    Args:
        :param qrels_path: Path to the qrels file.
        :param relevance_threshold: minimum relevance category for being considered relevant.

    Returns:
        dict: A dictionary where key=query_id, value=set of relevant doc IDs.
    """
    qrels = {}
    with open(qrels_path, 'r') as file:
        for line in file:
            query_id, _, doc_id, relevance = line.strip().split()
            if int(relevance) >= relevance_threshold:
                if query_id not in qrels:
                    qrels[query_id] = set()
                qrels[query_id].add(doc_id)
    return qrels


def calculate_relevant_ratio(query_id, retrieved_docs, qrels):
    """
    Calculate the ratio of relevant documents in the retrieved list.

    Args:
        query_id (str): Query ID.
        retrieved_docs (list): List of retrieved document IDs for the query.
        qrels (dict): Qrels dict where key=query_id, value=set of relevant doc IDs.

    Returns:
        float: Ratio of relevant documents in the retrieved list.
    """
    relevant_docs = qrels.get(query_id, set())  # Get relevant docs for the query
    retrieved_relevant_count = sum(1 for doc in retrieved_docs if doc.docid in relevant_docs)
    total_retrieved = len(retrieved_docs)

    return retrieved_relevant_count / total_retrieved if total_retrieved > 0 else 0.0


def precompute_ratios(parsed_results, qrels_path, relevance_threshold=2):
    """
    Precompute the ratio of relevant documents for all queries.

    Args:
        :param parsed_results: Parsed run file results (query_id -> list of SearchResult objects).
        :param qrels_path: Path to the qrels file.
        :param relevance_threshold: minimum relevance category for being considered relevant.

    Returns:
        dict: A dictionary where key=query_id, value=relevant ratio.
    """
    # Load qrels from the provided path
    qrels = load_qrels(qrels_path=qrels_path,relevance_threshold=relevance_threshold)

    ratios = {}
    for query_id, passages in parsed_results.items():
        ratios[query_id] = calculate_relevant_ratio(query_id, passages, qrels)
    return ratios


def cluster_parsed_results(parsed_results, clustering_method, n, qrels_path, ranker=None, relevance_threshold=2):
    """
    Cluster parsed run results and include relevant document ratio.

    Args:
        :param parsed_results: Parsed run file results (query_id -> list of SearchResult objects).
        :param clustering_method: Clustering method ("tfidf" or "llm").
        :param n: Number of nearest neighbors (NN) to retrieve per passage.
        :param qrels_path: Path to the qrels file.
        :param ranker: Optional, fine-tuned LLM ranker for "llm" clustering.
        :param relevance_threshold:

    Returns:
        dict: A dictionary where keys are query IDs and values are tuples
              (ratio of relevant docs, clustering results).
    """
    if clustering_method not in {"tfidf", "llm"}:
        raise ValueError(f"Unsupported clustering method: {clustering_method}")

    # Precompute relevant ratios
    relevant_ratios = precompute_ratios(parsed_results=parsed_results,
                                        qrels_path=qrels_path,
                                        relevance_threshold=relevance_threshold
                                        )

    clustering_results = {}
    num_of_queries = len(parsed_results)
    query_counter = 1
    for query_id, passages in parsed_results.items():
        if query_id not in relevant_ratios:
            print(f"Missing relevant_ratio for query_id: {query_id}")
        print(f"Processing query number {query_counter}/{num_of_queries}")
        query_counter += 1
        cluster_result = {}

        if clustering_method == "tfidf":
            # TF-IDF clustering
            cluster_result = retrieve_tfidf_nn(passages, n)
        elif clustering_method == "llm":
            if ranker is None:
                raise ValueError("An LLM ranker must be provided for 'llm' clustering.")
            # LLM clustering
            cluster_result = retrieve_llm_nn(passages, ranker, n)

        # Add ratio and cluster result to the final dictionary
        clustering_results[query_id] = (relevant_ratios[query_id], cluster_result)

    return clustering_results


def print_clusters(query_id, query_passages, clusters):
    """
    Print NN clusters for a specific query with details.

    Args:
        query_id (str): Query ID.
        query_passages (list): List of SearchResult objects for the query.
        clusters (dict): Clustering output for the query.
    """
    query_text = query_passages[0].text if query_passages else "No text available"
    print(f"Query ID: {query_id}")
    print(f"Query Text: {query_text[:100]}...\n")

    # Print each document and its NN cluster
    for doc_id, cluster in clusters.items():
        doc_text = next((p.text for p in query_passages if p.docid == doc_id), "No text found")
        print(f"DocID: {doc_id}, Text: {doc_text[:50]}...")
        print("  NN Cluster:")
        for result in cluster:
            print(f"    - DocID: {result.docid}, Text: {result.text[:50]}, Score: {result.score}")
        print()


def save_clustering_results_to_json(clustering_results, output_file):
    """
    Save clustering results to a JSON file as a tuple of (relevant_ratio, clusters).

    Args:
        clustering_results (dict): Clustering results where keys are query IDs
                                   and values are tuples (relevant_ratio, cluster dictionary of doc_id -> list of SearchResult objects).
        output_file (str): Path to the output JSON file.

    Returns:
        None
    """
    json_data = {}
    for query_id, (relevant_ratio, clusters) in clustering_results.items():
        json_data[query_id] = (
            relevant_ratio,
            {
                doc_id: [
                    {
                        "doc_id": result.docid,
                        "score": result.score,
                        "text": result.text
                    }
                    for result in nn_cluster
                ]
                for doc_id, nn_cluster in clusters.items()
            }
        )

    # Save to JSON
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, ensure_ascii=False, indent=4)
    print(f"Clustering results saved to {output_file}")


def load_clustering_results_from_json(clustering_file_path):
    with open(clustering_file_path, "r", encoding="utf-8") as f:
        clustering_results = json.load(f)
    return clustering_results


def create_and_save_all_clusters(bm25_path, llm_path, bm_tfids_out_path, llm_tfids_out_path, llm_llm_out_path, bm25_llm_out_path, ranker, docstore, qrels_path, cluster_size=5, ret_list_size=5, relevance_threshold=2):
    bm25_results = parse_run_file_with_search_results(bm25_path, ret_list_size=ret_list_size, docstore=docstore)
    llm_results = parse_run_file_with_search_results(llm_path, ret_list_size=ret_list_size, docstore=docstore)
    llm_tfidf_clusters = cluster_parsed_results(
        parsed_results=llm_results,
        clustering_method="tfidf",
        n=cluster_size,
        qrels_path=qrels_path,
        relevance_threshold=relevance_threshold
    )
    save_clustering_results_to_json(llm_tfidf_clusters, llm_tfids_out_path)

    llm_llm_clusters = cluster_parsed_results(
        parsed_results=llm_results,
        clustering_method="llm",
        n=cluster_size,
        ranker=ranker,
        qrels_path=qrels_path,
        relevance_threshold=relevance_threshold
    )
    save_clustering_results_to_json(llm_llm_clusters, llm_llm_out_path)

    bm25_tfidf_clusters = cluster_parsed_results(
        parsed_results=bm25_results,
        clustering_method="tfidf",
        n=cluster_size,
        qrels_path=qrels_path,
        relevance_threshold=relevance_threshold
    )
    save_clustering_results_to_json(bm25_tfidf_clusters, bm_tfids_out_path)

    bm25_llm_clusters = cluster_parsed_results(
        parsed_results=bm25_results,
        clustering_method="llm",
        n=cluster_size,
        ranker=ranker,
        qrels_path=qrels_path,
        relevance_threshold=relevance_threshold
    )
    save_clustering_results_to_json(bm25_llm_clusters, bm25_llm_out_path)


def main():
    pass

if __name__ == '__main__':
    main()
