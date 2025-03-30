from dataset import get_relevance_dict
from collections import Counter
import numpy as np
import json
import os

#hyper-params: how many selected docs from each region of the ranked list
T = 1000
M = 0
B = 0

def relevance_percentage(relevance_dict):
    # Count the occurrences of each relevance value
    relevance_counts = Counter(relevance_dict.values())

    # Total number of documents
    total_docs = len(relevance_dict)

    # Calculate the percentage for each relevance value
    rel_percentage = {
        key: (count / total_docs) * 100
        for key, count in relevance_counts.items()
    }

    return rel_percentage

def binary_relevance_percentage(relevance_dict, docs_ids, relevance_threshold=2):
    # Filter the relevance_dict to include only the documents in docs_ids
    filtered_relevance_dict = {doc_id: rel for doc_id, rel in relevance_dict.items() if doc_id in docs_ids}

    relevance_dict = {doc_id: ('relevant' if rel >= relevance_threshold else 'not relevant') for doc_id, rel in filtered_relevance_dict.items()}
    relevance_counts = Counter(relevance_dict.values())
    total_docs = len(relevance_dict)
    relevance_ratio = relevance_counts.get('relevant', 0) / total_docs if total_docs > 0 else 0
    num_relevant_docs = relevance_counts.get('relevant', 0)

    return num_relevant_docs, relevance_ratio



def eval_nn(docs_ids, GRR, relevance_dict):
  # calculate the number and percentage of relevant docs in nn
  nor, LRR = binary_relevance_percentage(relevance_dict, docs_ids)

  # calculate ratio of relevance compare to the ratio of relevance in the entire ranking
  RRR = LRR / GRR
  return nor, LRR, RRR


def eval_method(file_path, dataset_name, cluster_size, relevance_threshold=2):
  # get dictionary with query_id as keys, with value of dictionary of nn for each doc_id in most relevant for the query
  with open(file_path, "r") as file:
    query_docnn_dict = json.load(file)

  total_nor = np.zeros(cluster_size + 1)
  total_checks = 0
  sum_avg_LRR = 0
  sum_avg_RRR = 0

  for query_id in query_docnn_dict.keys():

    GRR, docnn_dict = query_docnn_dict[query_id]

    relevance_dict = get_relevance_dict(query_id, list(docnn_dict.keys()), dataset_name)
    relevant_docs = [k for k,v in relevance_dict.items() if v>=relevance_threshold]

    sum_LRR = 0
    sum_RRR = 0

    for doc_id in docnn_dict.keys():
        
      if doc_id in relevant_docs:
        doc_ids = [doc['doc_id'] for doc in docnn_dict[doc_id]]
    
        # eval the list of nn of doc_id for each method
        nor, LRR, RRR = eval_nn(doc_ids , GRR, relevance_dict)

        total_nor[nor] += 1
        sum_LRR += LRR
        sum_RRR += RRR
     
    if len(relevant_docs) > 0 :
      total_checks += len(relevant_docs)
      sum_avg_LRR += (sum_LRR/len(relevant_docs))
      sum_avg_RRR += (sum_RRR/len(relevant_docs))

  pnor = total_nor/total_checks
  mean_avg_LRR = sum_avg_LRR / len(query_docnn_dict.keys())
  mean_avg_RRR = sum_avg_RRR / len(query_docnn_dict.keys())

  return pnor, mean_avg_LRR, mean_avg_RRR


def write_evaluation_measures(eval_file, cluster_files, dataset_name, cluster_size, relevance_threshold):
    """
    Parametrized function to write evaluation measures to a file.

    Args:
        :param relevance_threshold: threshold for a doc to be considered relevant
        :param cluster_size: size of the clusters
        :param eval_file: The name of the evaluation file to write.
        :param cluster_files: A dictionary mapping measure labels to cluster file names.
        :param dataset_name: name of ir_datasets dataset
    """


    measure_labels = ["PNOR", "Mean Average LRR", "Mean Average RRR"]

    if os.path.exists(eval_file):
        os.remove(eval_file)

    with open(eval_file, "w") as _:
        pass

    # Compute and write measures for each cluster file
    with open(eval_file, "w") as file:
        for label, cluster_file in cluster_files.items():
            measures = eval_method(cluster_file, dataset_name, cluster_size, relevance_threshold)  # Evaluate the measures
            file.write(f"=== {label} ===\n")
            for i, measure in enumerate(measures):
                file.write(f"{measure_labels[i]}: {measure}\n")
            file.write("\n")


def main():
  pass

if __name__ == "__main__":
    main()