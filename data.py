import pandas as pd
import json
import os

from config import STORAGE_DIR


def get_mtrag_queries(path):
    if not path.endswith(".jsonl"):
        raise ValueError("Path must end with .jsonl")
    ids = []
    queries = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            ids.append(str(obj["_id"]))
            queries.append(str(obj["text"]))
    return ids, queries


def get_legal_dataset(path):
    """
    Load the legal dataset from a JSON or CSV file.
    
    Args:
        path (str): Path to the JSON or CSV file.
    
    Returns:
        tuple: A tuple containing two lists:
            - List of document IDs (Codigo).
            - List of document texts (text).
    """
    fixed_lines = []
    # Load the dataset
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                json.loads(line)
                fixed_lines.append(line)
            except json.JSONDecodeError:
                pass
        corpus_dict = [json.loads(line) for line in fixed_lines]
        print(f"Loaded {len(corpus_dict)} records from {path}")
    df = pd.DataFrame(corpus_dict)

    # convert Codigo column datatype to str
    df["id"] = df["id"].astype(str)

    doc_ids = df["id"].tolist()
    docs = df["text"].tolist()

    return doc_ids, docs


class Data:
    def __init__(self, qtype, corpus) -> None:
        """ qtype is either rewrite or lastturn.
            corpus is either clapnq, cloud, fiqa or govt
        """
        self.doc_ids = []
        self.docs = []

        self.query_ids = []
        self.queries = []

        self.doc_dict = {}
        self.query_dict = {}
        self.qrels_dev_df = {}

        self.load_data(qtype, corpus)

    def load_data(self, qtype, corpus):        
        # load corpus
        self.doc_ids, self.docs = get_legal_dataset(os.path.join(STORAGE_DIR, "mtrag", "data", "corpus", f"{corpus}.jsonl"))
        self.doc_dict = {doc_id: doc for doc_id, doc in zip(self.doc_ids, self.docs)}

        # load queries
        queries_path = os.path.join(STORAGE_DIR, "mtrag", "data", "questions", f"{corpus}_{qtype}.jsonl")
        self.query_ids, self.queries = get_mtrag_queries(queries_path)
        # create a dictionary of query_id to query
        self.query_dict = {query_id: query for query_id, query in zip(self.query_ids, self.queries)}

        # load qrels
        # query-id	corpus-id	score
        path_to_reference_qrels = os.path.join(STORAGE_DIR, "mtrag", "data", "annotations", f"{corpus}_qrels.tsv")

        self.qrels_dev_df = pd.read_csv(
            path_to_reference_qrels,
            sep="\t",                # TREC qrels are usually tab-separated
            names=["query_id", "doc_id", "relevance"],
            header=0,            # There's no header in qrels files
            dtype={"query_id": str, "doc_id": str, "relevance": int}
        )

