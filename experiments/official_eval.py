"""Official AraSeg-2026 evaluation script — vendored verbatim from
https://github.com/mbzuai-nlp/araseg-shared-task-2026/blob/main/scripts/eval.py

This is the authoritative scorer for our written submission CSVs. Our in-repo
metrics.py is algebraically identical (per-doc sklearn average='binary', then
np.mean), but we run THIS on the actual CSV so local numbers are leaderboard-
exact, not just equivalent.

ONLY change vs upstream: the two final print lines below. Upstream prints
macro_recall under the "Precision:" label and macro_precision under "Recall:"
(a label swap; the values are computed correctly). Fixed here so we don't make
submission decisions off mislabeled P/R.
"""
from argparse import ArgumentParser
from typing import Dict, FrozenSet, List

import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.metrics import f1_score, precision_score, recall_score

TASK_TO_DATASET_NAMES: FrozenSet[str] = {
    "PA": "MBZUAI/AraSeg-2026-Shared-Task-PA",
    "NP": "MBZUAI/AraSeg-2026-Shared-Task-NP",
    "NoPnx-PA": "MBZUAI/AraSeg-2026-Shared-Task-NoPnx-PA",
    "NoPnx-NP": "MBZUAI/AraSeg-2026-Shared-Task-NoPnx-NP"
}

SPLITS: FrozenSet[str] = frozenset(["train", "dev", "test"])

def load_data_from_hf(task_name, split=None):
    assert task_name in TASK_TO_DATASET_NAMES, f"Task {task_name} not found in the dataset names"
    assert split in SPLITS, f"Split {split} not found in the splits"
    if split is None:
        dataset = load_dataset(TASK_TO_DATASET_NAMES[task_name])
        return dataset
    else:
        dataset = load_dataset(TASK_TO_DATASET_NAMES[task_name], split=split)
        return dataset

def convert_binary_string_to_list(binary_string: str) -> List[int]:
    return [int(char) for char in binary_string]

def load_predictions(prediction_file_path: str) -> Dict[str, List[int]]:
    df = pd.read_csv(prediction_file_path, header=0)
    docs = {}
    for index, row in df.iterrows():
        doc_id = row["Document ID"]
        labels = convert_binary_string_to_list(row["Prediction"])
        docs[doc_id] = labels
    return docs

def compute_metrics(gold_documents: Dict[str, List[int]], predicted_documents: Dict[str, List[int]]):
    assert len(gold_documents) == len(predicted_documents), "Gold and predicted documents must have the same length"
    doc_metrics = {}
    for doc in gold_documents:
        gold_labels = gold_documents[doc]
        pred_labels = predicted_documents[doc]
        assert len(gold_labels) == len(pred_labels), "Gold and predicted labels must have the same length"
        doc_metrics[doc] = {
            'precision': precision_score(gold_labels, pred_labels, average='binary'),
            'recall': recall_score(gold_labels, pred_labels, average='binary'),
            'f1': f1_score(gold_labels, pred_labels, average='binary')
        }
    macro_precision = np.mean([metric['precision'] for metric in doc_metrics.values()])
    macro_recall = np.mean([metric['recall'] for metric in doc_metrics.values()])
    macro_f1 = np.mean([metric['f1'] for metric in doc_metrics.values()])
    return {
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'macro_f1': macro_f1
    }

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--task", type=str, required=True, choices=TASK_TO_DATASET_NAMES.keys(), help="The name of the task to evaluate")
    parser.add_argument("--predictions", type=str, required=True, help="The path to the prediction file")
    parser.add_argument("--split", type=str, required=False, default="dev", choices=SPLITS, help="The split to evaluate on")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    task_name = args.task
    prediction_file_path = args.predictions
    split = args.split

    print("-"*20)
    print(f"Prediction file path: {prediction_file_path}")
    print(f"Task name: {task_name}")
    print(f"Split: {split}")
    print("-"*20)

    # load data
    data = load_data_from_hf(task_name, split)
    gold_documents = {doc["doc_id"]: doc["labels"] for doc in data}
    predicted_documents = load_predictions(prediction_file_path)
    assert set(gold_documents.keys()) == set(predicted_documents.keys()), f"Gold and predicted documents must have the same keys. Missing gold documents: {set(gold_documents.keys()) - set(predicted_documents.keys())}"
    metrics = compute_metrics(gold_documents, predicted_documents)
    metrics = {
        "macro_precision": metrics["macro_precision"].item(),
        "macro_recall": metrics["macro_recall"].item(),
        "macro_f1": metrics["macro_f1"].item()
    }
    # NOTE: upstream eval.py swaps these two labels (prints macro_recall as
    # "Precision" and macro_precision as "Recall"). Corrected here.
    print(f'Precision: {metrics["macro_precision"]*100:.4}')
    print(f'Recall:    {metrics["macro_recall"]*100:.4}')
    print(f'F1:        {metrics["macro_f1"]*100:.4}')

    print("-"*20)
