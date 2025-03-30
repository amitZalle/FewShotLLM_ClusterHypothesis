import pytrec_eval

def load_trec_qrels(qrels_file):
    """Load qrels (ground truth) from a TREC-formatted qrels file."""
    qrels = {}
    with open(qrels_file, "r") as f:
        for line in f:
            qid, _, doc_id, relevance = line.strip().split()
            if qid not in qrels:
                qrels[qid] = {}
            qrels[qid][doc_id] = int(relevance)  # Relevance scores are stored but treated as binary for recall
    return qrels

def load_trec_results(results_file):
    """Load ranked retrieval results from a TREC-formatted results file."""
    results = {}
    with open(results_file, "r") as f:
        for line in f:
            qid, _, doc_id, rank, score, _ = line.strip().split()
            if qid not in results:
                results[qid] = {}
            results[qid][doc_id] = float(score)  # Store scores
    return results


def compute_metrics(qrels_path, bm25_results_path, llm_results_path, output_file):
    """Compute Recall@100 (shared), MAP, and NDCG@10 for BM25 and LLM results."""
    qrels = load_trec_qrels(qrels_path)
    bm25_results = load_trec_results(bm25_results_path)
    llm_results = load_trec_results(llm_results_path)

    evaluator = pytrec_eval.RelevanceEvaluator(qrels, {'map', 'ndcg_cut_10', 'recall_100'})

    # Evaluate BM25
    bm25_scores = evaluator.evaluate(bm25_results)
    map_bm25 = sum(q['map'] for q in bm25_scores.values()) / len(bm25_scores)
    ndcg_bm25 = sum(q['ndcg_cut_10'] for q in bm25_scores.values()) / len(bm25_scores)
    recall_100 = sum(q['recall_100'] for q in bm25_scores.values()) / len(bm25_scores)  # Shared recall

    # Evaluate LLM
    llm_scores = evaluator.evaluate(llm_results)
    map_llm = sum(q['map'] for q in llm_scores.values()) / len(llm_scores)
    ndcg_llm = sum(q['ndcg_cut_10'] for q in llm_scores.values()) / len(llm_scores)

    # Write results to file
    with open(output_file, "w") as f:
        f.write(f"Recall@100: {recall_100:.4f}\n\n")

        f.write("BM25:\n")
        f.write(f"MAP: {map_bm25:.4f}\n")
        f.write(f"NDCG@10: {ndcg_bm25:.4f}\n\n")

        f.write("LLM:\n")
        f.write(f"MAP: {map_llm:.4f}\n")
        f.write(f"NDCG@10: {ndcg_llm:.4f}\n")

    print(f"Evaluation results saved to {output_file}")


if __name__ == "__main__":
    pass