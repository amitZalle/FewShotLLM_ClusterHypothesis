import ir_datasets

def get_dataset(dataset_name):
  # Load the dataset
  dataset = ir_datasets.load(dataset_name)
  return dataset


def get_docs_text(doc_ids, dataset_name):
  dataset = get_dataset(dataset_name)

  doc_texts = {}
  
  for doc_id in doc_ids:
      doc = dataset.docs[doc_id]
      doc_texts[doc_id] = doc.text
      
  return doc_texts


def get_relevance_dict(query_id, doc_ids, dataset_name):
  dataset = get_dataset(dataset_name)

  relevance_dict = {doc_id: 0 for doc_id in doc_ids}  # Default relevance 0

  for qrel in dataset.qrels_iter():
      if qrel.query_id == query_id and qrel.doc_id in doc_ids:
          relevance_dict[qrel.doc_id] = qrel.relevance  # Update with actual relevance score
          
  return relevance_dict