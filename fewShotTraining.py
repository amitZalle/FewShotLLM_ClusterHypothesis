from transformers import TrainingArguments, Trainer
from torch.utils.data import Dataset
from collections import defaultdict
import random


def safe_sample(lst, k):
    """
    Sample k items from lst.
    If k > len(lst), sample with replacement.
    If k <= len(lst), sample without replacement.
    """
    if len(lst) >= k:
        return random.sample(lst, k)
    else:
        return random.choices(lst, k=k)


def generate_training_examples(qrels_paths, queries_list, docstores, examples_per_query=6, relevance_threshold=2, method="pointwise", sub_method=None):
    from collections import defaultdict
    query_groups = defaultdict(lambda: {"positive": [], "negative": []})

    for qrels_path, queries, docstore in zip(qrels_paths, queries_list, docstores):
        with open(qrels_path, 'r') as f:
            for line in f:
                query_id, _, doc_id, relevance = line.strip().split()
                relevance = int(relevance)
                if query_id not in queries:
                    continue
                query = queries[query_id]
                doc_text = docstore.get(doc_id).text
                if relevance >= relevance_threshold:
                    query_groups[query]["positive"].append((doc_text, relevance))
                else:
                    query_groups[query]["negative"].append((doc_text, relevance))

    examples = []

    for query, bins in query_groups.items():
        pos = bins["positive"]
        neg = bins["negative"]

        if method == "pointwise":
            n_pos = examples_per_query // 2
            n_neg = examples_per_query - n_pos
            selected_pos = safe_sample(pos, n_pos)
            selected_neg = safe_sample(neg, n_neg)

            for doc, _ in selected_pos:
                examples.append({
                    "input": f"Passage: {doc}\nQuery: {query}\nDoes the passage answer the query? Answer 'Yes' or 'No'",
                    "label": "Yes"
                })
            for doc, _ in selected_neg:
                examples.append({
                    "input": f"Passage: {doc}\nQuery: {query}\nDoes the passage answer the query? Answer 'Yes' or 'No'",
                    "label": "No"
                })

        elif method == "pairwise":
            half = examples_per_query // 2
            pos1 = safe_sample(pos, half)
            neg1 = safe_sample(neg, half)
            neg2 = safe_sample(neg, half)
            pos2 = safe_sample(pos, half)

            for (p, _), (n, _) in zip(pos1, neg1):
                examples.append({
                    "input": f'Given the following query: "{query}"\nPassage A: "{p}"\nPassage B: "{n}"\nWhich passage is more relevant?',
                    "label": "Passage A"
                })

            for (n, _), (p, _) in zip(neg2, pos2):
                examples.append({
                    "input": f'Given the following query: "{query}"\nPassage A: "{n}"\nPassage B: "{p}"\nWhich passage is more relevant?',
                    "label": "Passage B"
                })

        elif method == "listwise":
            if sub_method not in {"generation", "likelihood"}:
                raise ValueError("Listwise requires sub_method: 'generation' or 'likelihood'")

            third = examples_per_query // 3
            for _ in range(third):
                group = (
                    safe_sample(pos, 2) + safe_sample(neg, 2)  # pos-pos-neg-neg
                )
                examples.append(_build_listwise_example(query, group, sub_method))

            for _ in range(third):
                group = (
                    safe_sample(pos, 1) + safe_sample(neg, 3)  # pos-neg-neg-neg
                )
                examples.append(_build_listwise_example(query, group, sub_method))

            for _ in range(third):
                group = (
                    safe_sample(pos, 3) + safe_sample(neg, 1)  # pos-pos-pos-neg
                )
                examples.append(_build_listwise_example(query, group, sub_method))
        else:
            raise ValueError(f"Unsupported method: {method}")

    return examples


def _build_listwise_example(query, group, sub_method):
    """
    Helper to construct a listwise training example.
    """
    random.shuffle(group)
    indexed = list(enumerate(group, start=1))

    if sub_method == "generation":
        input_text = (
            "### INSTRUCTIONS ###\n"
            "You are RankGPT, an expert ranking assistant.\n\n"
            "TASK: Rank the following passages in **descending order of relevance** to the query.\n\n"
            f"QUERY: {query}\n"
            "---PASSAGES---\n" +
            "\n".join([f"[{i}] {doc}" for i, (doc, _) in indexed]) +
            "\n\n### RANKING FORMAT ###\n"
            "- List all passage identifiers in **descending order of relevance**.\n"
            "- Use the exact format: [3] > [1] > [2]\n"
            "- Make sure to rank all passages"
        )
        sorted_labels = sorted(indexed, key=lambda x: -x[1][1])
        label_text = " > ".join([f"[{i}]" for i, _ in sorted_labels])
        return {"input": input_text, "label": label_text}

    elif sub_method == "likelihood":
        input_text = (
            "### INSTRUCTIONS ###\n"
            "You are RankGPT, an expert passage evaluator.\n\n"
            "TASK: Select the **single most relevant passage** to the following query.\n\n"
            f"QUERY: {query}\n\n"
            "--- PASSAGES ---\n" +
            "\n".join([f"[{i}] {doc}" for i, (doc, _) in indexed]) +
            "\n\n### OUTPUT FORMAT ###\n"
            "- Output the **label** of the most relevant passage only.\n"
            "- Use the exact format: [1]"
        )
        top_doc = max(group, key=lambda x: x[1])
        for i, (doc, _) in indexed:
            if doc == top_doc[0]:
                return {"input": input_text, "label": f"[{i}]"}
    else:
        raise ValueError(f"Unknown sub_method: {sub_method}")


def tokenize_data(examples, tokenizer, max_length=512):
    """
    Tokenize input-label pairs for T5 training.

    Args:
        examples (list): List of dictionaries with 'input' and 'label'.
        tokenizer: T5 tokenizer.
        max_length (int): Maximum token length.

    Returns:
        dict: Tokenized input and labels.
    """
    inputs = [ex['input'] for ex in examples]
    labels = [ex['label'] for ex in examples]

    tokenized_inputs = tokenizer(
        inputs, padding=True, truncation=True, max_length=max_length, return_tensors="pt"
    )
    tokenized_labels = tokenizer(
        labels, padding=True, truncation=True, max_length=16, return_tensors="pt"
    )

    tokenized_inputs["labels"] = tokenized_labels["input_ids"]
    return tokenized_inputs


def test_tokenize_data(examples, tokenizer, max_length=512):
    """
    Test the tokenize_data function by printing tokenized inputs and labels.

    Args:
        examples (list): List of dictionaries with 'query', 'document', and 'label'.
        tokenizer: T5 tokenizer.
        max_length (int): Maximum token length.
    """
    print("==== Tokenize Data Test ====")
    tokenized_data = tokenize_data(examples, tokenizer, max_length)

    # Iterate through tokenized examples and print their details
    for i, example in enumerate(examples):
        print(f"Example {i + 1}:")
        print(f"  Query: {example['query']}")
        print(f"  Document: {example['document'][:100]}...")  # Truncate for readability
        print(f"  Label: {example['label']}")

        # Decode tokenized inputs for verification
        decoded_input = tokenizer.decode(tokenized_data["input_ids"][i], skip_special_tokens=True)

        print(f"  Tokenized Input: {decoded_input}")
        print(f"  Tokenized Label: {example['label']}")  # Print label directly
        print("-" * 50)


class FewShotDataset(Dataset):
    def __init__(self, tokenized_data):
        """
        Initialize the dataset with tokenized data.
        Args:
            tokenized_data (dict): A dictionary containing tokenized inputs and labels.
        """
        self.data = tokenized_data

    def __len__(self):
        """
        Return the total number of examples in the dataset.
        """
        return len(self.data["input_ids"])

    def __getitem__(self, idx):
        """
        Retrieve an example at the given index.
        """
        return {key: val[idx] for key, val in self.data.items()}

    def print_sample(self, num_samples=5):
        """
        Print a few samples from the dataset for inspection.
        Args:
            num_samples (int): Number of samples to print.
        """
        print(f"Dataset length: {len(self)}")
        print("==== Sample Data ====")
        for i in range(min(num_samples, len(self))):
            print(f"Example {i + 1}:")
            for key, val in self[i].items():
                print(f"  {key}: {val}")
            print("-" * 50)


def train_model(model, train_dataset, tokenizer, output_path):
    training_args = TrainingArguments(
        output_dir=output_path,
        num_train_epochs=3,
        per_device_train_batch_size=8,
        warmup_steps=500,
        logging_dir="./logs",
        logging_steps=10,
        save_steps=500,
        save_total_limit=2,
        learning_rate=0.0001,
        weight_decay=0.01,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
    )

    trainer.train()

    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)

def print_training_examples(examples, limit_per_query=5):
    """
    Print the training examples in a readable format, grouped by query,
    with a check for queries that lack positives or negatives.

    Args:
        examples (list): List of training examples (query, document, label).
        limit_per_query (int): Number of examples (per type) to print for each query.
    """
    print("==== Training Examples Grouped by Query ====")
    grouped_examples = {}

    # Group examples by query
    for example in examples:
        query = example['query']
        if query not in grouped_examples:
            grouped_examples[query] = []
        grouped_examples[query].append(example)

    # Print examples grouped by query
    for i, (query, query_examples) in enumerate(grouped_examples.items()):
        print(f"\nQuery {i + 1}: {query}")
        print("-" * 50)

        # Separate positives and negatives
        positives = [ex for ex in query_examples if ex['label'] == 1][:limit_per_query]
        negatives = [ex for ex in query_examples if ex['label'] == 0][:limit_per_query]

        # Handle cases with no positives or negatives
        if not positives:
            print("No Positive Examples for this Query.")
        else:
            print(f"Positive Examples (limit {limit_per_query}):")
            for ex in positives:
                print(f"  Document: {ex['document'][:100]}...")  # Truncate for readability
                print(f"  Label: {ex['label']}")
                print("  ---")

        if not negatives:
            print("No Negative Examples for this Query.")
        else:
            print(f"\nNegative Examples (limit {limit_per_query}):")
            for ex in negatives:
                print(f"  Document: {ex['document'][:100]}...")  # Truncate for readability
                print(f"  Label: {ex['label']}")
                print("  ---")