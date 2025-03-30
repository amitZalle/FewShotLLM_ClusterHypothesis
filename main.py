from transformers import T5ForConditionalGeneration, T5Tokenizer
from eval import write_evaluation_measures
from pointwise import PointwiseLlmRanker
from pairwise import PairwiseLlmRanker
from listwise import ListwiseLlmRanker
from run import main as run_main
from argparse import Namespace
from datetime import datetime
import fewShotTraining
import retrieval_eval
import bm25Retriever
import ir_datasets
import clustering
import torch
import json
import os

# Device configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.cuda.empty_cache()

# Model config
RANKER_METHOD = "pointwise" # Options - pointwise, pairwise, listwise
SUB_METHOD = "yes_no" # pointwise - (yes_no) , pairwise - (allpair, heapsort, bubblesort), listwise - (generation, likelihood)
MODEL_SIZE = "small" # small/base/large
IS_FLAN = True

# Training params
NUM_EXAMPLES_PER_QUERY = 6
RELEVANCE_THRESHOLD = 2

# Batch size for reranking
BATCH_SIZE = 4

# List size params
INITIAL_LIST_SIZE=100
FINAL_LIST_SIZE=20
CLUSTER_SIZE = 5


if IS_FLAN:
    BASE_MODEL_NAME = f"google/flan-t5-{MODEL_SIZE}" # Base model for fine-tuning
    RUN_DIR = f"./run_logs/flan_{MODEL_SIZE}_{RANKER_METHOD}_{SUB_METHOD}/{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    TRAINED_MODEL_PATH = f"{RUN_DIR}/fine_tuned_flan_t5_{MODEL_SIZE}_{RANKER_METHOD}_{SUB_METHOD}"
else:
    BASE_MODEL_NAME = f"google/t5-{MODEL_SIZE}"  # Base model for fine-tuning
    RUN_DIR = f"./run_logs/{MODEL_SIZE}_{RANKER_METHOD}_{SUB_METHOD}/{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    TRAINED_MODEL_PATH = f"{RUN_DIR}/fine_tuned_t5_{MODEL_SIZE}_{RANKER_METHOD}_{SUB_METHOD}"

os.makedirs(RUN_DIR, exist_ok=True)

# Define dataset and model constants
TRAIN_DATASET_NAME = "msmarco-passage/trec-dl-2020/judged" # Dataset for few-shot training
TEST_DATASET_NAME = "msmarco-passage/trec-dl-2019/judged"
INDEX_NAME = "msmarco-passage" # Retrieval index

# Output file paths (all within the RUN_DIR)
BM25_FILE = os.path.join(RUN_DIR, "bm25_run.txt")
LLM_FILE = os.path.join(RUN_DIR, "llm_run.txt")
RETRIEVAL_EVAL_FILE = os.path.join(RUN_DIR, "retrieval_eval.txt")
PARAMETERS_LOG_FILE = os.path.join(RUN_DIR, "parameters.json")
QUERIES_LOG_FILE = os.path.join(RUN_DIR, "queries.json")
TRAINING_DATASET_FILE = os.path.join(RUN_DIR, "training_dataset.json")

# Clustering output paths
BM25_TFIDF_OUT_PATH = os.path.join(RUN_DIR, "bm25_tfidf_clusters.json")
LLM_TFIDF_OUT_PATH = os.path.join(RUN_DIR, "llm_tfidf_clusters.json")
LLM_LLM_OUT_PATH = os.path.join(RUN_DIR, "llm_llm_clusters.json")
BM25_LLM_OUT_PATH = os.path.join(RUN_DIR, "bm25_llm_clusters.json")
EVAL_OUT_PATH = os.path.join(RUN_DIR, "evaluation_measures.txt")


def rerank_bm25_from_py(run_path, output_path, trained_model_path, ranker_method, device="cuda", pyserini_index=None, ir_dataset_name=None):
    # Ensure the model path is absolute
    trained_model_path = os.path.abspath(trained_model_path)
    print(f"Passing model path: {trained_model_path}")  # Debugging statement

    # Create args structure expected by `run.py`
    args = Namespace(
        run=Namespace(
            run_path=run_path,
            save_path=output_path,
            model_name_or_path=trained_model_path,
            tokenizer_name_or_path=None,
            ir_dataset_name=ir_dataset_name,
            pyserini_index=pyserini_index,
            hits=INITIAL_LIST_SIZE,
            query_length=128,
            passage_length=128,
            device=device,
            cache_dir=None,
            openai_key=None,
            shuffle_ranking=None,
            scoring=SUB_METHOD
        ),
        pointwise=None,
        pairwise=None,
        setwise=None,
        listwise=None,
    )

    if ranker_method == "pointwise":
        args.pointwise = Namespace(
            method=SUB_METHOD,
            batch_size=BATCH_SIZE,
        )
    elif ranker_method == "pairwise":
        args.pairwise = Namespace(
            method=SUB_METHOD,
            batch_size=BATCH_SIZE,
            k=10,
        )
    elif ranker_method == 'listwise':
        args.listwise = Namespace(
            window_size=3,
            step_size=1,
            num_repeat=1
        )
    else:
        raise ValueError(f"Invalid ranker method: {ranker_method}")

    try:
        run_main(args)
    except Exception as e:
        raise RuntimeError(f"Error occurred during reranking: {e}")



def save_parameters_and_queries(parameters, queries):
    """
    Save parameters and queries to log files in the RUN_DIR.

    Args:
        parameters (dict): Dictionary of parameters.
        queries (dict): Dictionary of query_id -> query_text.
    """
    with open(PARAMETERS_LOG_FILE, "w") as f:
        json.dump(parameters, f, indent=4)
    with open(QUERIES_LOG_FILE, "w") as f:
        json.dump(queries, f, indent=4)
    print(f"Step 0 - Save parameters and queries: COMPLETED")


def save_training_dataset(examples):
    """
    Save the training dataset (query-document-label pairs) to a JSON file.

    Args:
        examples (list): List of dictionaries with 'query', 'document', and 'label'.
    """
    with open(TRAINING_DATASET_FILE, "w") as f:
        json.dump(examples, f, indent=4)


def main():
    # Step 0: Save Parameters and Queries
    dataset = ir_datasets.load(TEST_DATASET_NAME)
    docstore = dataset.docs_store()
    qrels_path = dataset.qrels_path()

    train_dataset = ir_datasets.load(TRAIN_DATASET_NAME)
    train_docstore = train_dataset.docs_store()
    train_qrels_path = train_dataset.qrels_path()

    parameters = {
        "test_dataset": TEST_DATASET_NAME,
        "train_dataset": TRAIN_DATASET_NAME,
        "index": INDEX_NAME,
        "base_model": BASE_MODEL_NAME,
        "trained_model_path": TRAINED_MODEL_PATH,
        "examples_per_query": NUM_EXAMPLES_PER_QUERY,
        "relevance_threshold": RELEVANCE_THRESHOLD,
        "initial_list_size": INITIAL_LIST_SIZE,
        "final_list_size": FINAL_LIST_SIZE,
        "cluster_size": CLUSTER_SIZE,
        "ranker_method": RANKER_METHOD,
        "sub_method": SUB_METHOD,
    }
    queries = {q.query_id: q.text for q in train_dataset.queries_iter()}
    save_parameters_and_queries(parameters, queries)

    # Step 1: Generate BM25 results
    bm25Retriever.generate_bm25_run_file(
        dataset_name=TEST_DATASET_NAME,
        index_name=INDEX_NAME,
        output_file=BM25_FILE,
        top_k=INITIAL_LIST_SIZE
    )
    print("Step 1 - Generate BM25 results: COMPLETED")

    # Step 2: Few-shot dataset creation
    train_data = fewShotTraining.generate_training_examples(
        qrels_paths=[train_qrels_path],
        queries_list=[queries],
        docstores=[train_docstore],
        examples_per_query=NUM_EXAMPLES_PER_QUERY,
        relevance_threshold=RELEVANCE_THRESHOLD,
        method="pointwise",
        sub_method="yes_no"
    )
    tokenized_data = fewShotTraining.tokenize_data(train_data, T5Tokenizer.from_pretrained(BASE_MODEL_NAME))
    train_dataset = fewShotTraining.FewShotDataset(tokenized_data)

    # Save the entire training dataset
    save_training_dataset(train_data)
    print("Step 2 - Build dataset: COMPLETED")

    # Step 3: Few-shot training
    model = T5ForConditionalGeneration.from_pretrained(BASE_MODEL_NAME)

    fewShotTraining.train_model(
        model,
        train_dataset,
        T5Tokenizer.from_pretrained(BASE_MODEL_NAME),
        TRAINED_MODEL_PATH
    )
    print("Step 3 - Train model: COMPLETED")

    # Step 4: Rerank BM25 results
    rerank_bm25_from_py(
        run_path=BM25_FILE,
        output_path=LLM_FILE,
        trained_model_path=TRAINED_MODEL_PATH,
        ranker_method=RANKER_METHOD,
        device=DEVICE,
        ir_dataset_name=TEST_DATASET_NAME
    )
    retrieval_eval.compute_metrics(qrels_path, BM25_FILE, LLM_FILE, RETRIEVAL_EVAL_FILE)
    print("Step 4 - Rerank BM25 results: COMPLETED")

    if RANKER_METHOD == 'pointwise':
        ranker = PointwiseLlmRanker(
            model_name_or_path=TRAINED_MODEL_PATH,
            tokenizer_name_or_path=BASE_MODEL_NAME,
            device=DEVICE,
            method=SUB_METHOD,
            mission='clustering'
        )
    elif RANKER_METHOD == 'pairwise':
        ranker = PairwiseLlmRanker(
            model_name_or_path=TRAINED_MODEL_PATH,
            tokenizer_name_or_path=BASE_MODEL_NAME,
            device=DEVICE,
            method=SUB_METHOD,
            k=FINAL_LIST_SIZE,
            mission='clustering'
        )
    elif RANKER_METHOD == 'listwise':
        ranker = ListwiseLlmRanker(
            model_name_or_path=TRAINED_MODEL_PATH,
            tokenizer_name_or_path=BASE_MODEL_NAME,
            device=DEVICE,
            window_size=10,
            step_size=5,
            scoring=SUB_METHOD,
            mission='clustering'
        )
    else:
        raise RuntimeError(f'no such method {RANKER_METHOD}')

    # Step 5: Cluster results and save to JSON files
    clustering.create_and_save_all_clusters(
        bm25_path=BM25_FILE,
        llm_path=LLM_FILE,
        bm_tfids_out_path=BM25_TFIDF_OUT_PATH,
        llm_tfids_out_path=LLM_TFIDF_OUT_PATH,
        llm_llm_out_path=LLM_LLM_OUT_PATH,
        bm25_llm_out_path=BM25_LLM_OUT_PATH,
        ranker=ranker,
        docstore=docstore,
        cluster_size=CLUSTER_SIZE,
        qrels_path=qrels_path,
        ret_list_size=FINAL_LIST_SIZE,
        relevance_threshold=RELEVANCE_THRESHOLD
    )
    print("Step 5 - Cluster results: COMPLETED")

    # Step 6: Cluster Evaluation
    cluster_files = {
        "BM25_TFIDF": BM25_TFIDF_OUT_PATH,
        "BM25_LLM": BM25_LLM_OUT_PATH,
        "LLM_TFIDF": LLM_TFIDF_OUT_PATH,
        "LLM_LLM": LLM_LLM_OUT_PATH
    }

    write_evaluation_measures(
        eval_file=EVAL_OUT_PATH,
        cluster_files=cluster_files,
        dataset_name=TEST_DATASET_NAME,
        cluster_size=CLUSTER_SIZE,
        relevance_threshold=RELEVANCE_THRESHOLD
    )
    print("Step 6 - Cluster Eval: COMPLETED")

if __name__ == "__main__":
    main()
