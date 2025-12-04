import os
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from config import STORAGE_DIR

MAX_QUERY_LEN = 48
MAX_DOC_LEN = 512


class Retriever:
    def __init__(self, model, doc_ids, docs, emb_path) -> None:
        self.model = model
        self.doc_ids = doc_ids
        self.docs = docs
        self.doc_dict = dict(zip(doc_ids, docs))
        self.embed_bge(docs, reuse_embeddings=True, path=emb_path)

    def compute_similarity(self, embeddings_queries, embeddings_docs, sim_type='dot'):
        """
        Given query and document embeddings, compute the similarity between queries and documents
        and return the similarity matrix.
        Dot-product and cosine similarity are supported.

        Parameters
        ----------
        embeddings_queries : torch.Tensor
            Tensor of query embeddings. Can be either 2D [num_queries, emb_dim] or 1D [emb_dim].
        embeddings_docs : torch.Tensor
            Tensor of document embeddings of shape [num_docs, emb_dim].
        sim_type : str, optional
            Type of similarity to use. Options: 'dot', 'cosine'. (default is 'dot').

        Returns
        -------
        torch.Tensor
            Similarity matrix of shape [num_queries, num_docs].
        """
        print("Computing similarity...", end="")

        # If there is just one query, unsqueeze to make it 2D
        if embeddings_queries.dim() == 1:
            embeddings_queries = embeddings_queries.unsqueeze(0)

        if sim_type == 'dot':
            similarities = []
            batch_size = 32
            for query_batch in torch.split(embeddings_queries, batch_size, dim=0):
                sim_batch = query_batch @ embeddings_docs.T
                similarities.append(sim_batch)
            similarity = torch.cat(similarities, dim=0)
        elif sim_type == 'cosine':
            # Normalize embeddings along the embedding dimension
            embeddings_queries = F.normalize(embeddings_queries, p=2, dim=1)
            embeddings_docs = F.normalize(embeddings_docs, p=2, dim=1)
            similarity = (embeddings_queries @ embeddings_docs.T) * 100
        else:
            raise ValueError(f"Invalid similarity type: {sim_type}")
        
        print("Done.")
        return similarity

    def _retrieve(self, query_ids, similarity, top_k=None):
        """
        Build a run dictionary:
            {
                query_id: {doc_id: score, ...}
            }

        Parameters
        ----------
        query_ids : list
        similarity : 2D array-like where similarity[i][j] is score for query i and doc j
        top_k : int or None
            If set, return only the top_k most similar docs per query.
        """
        run = {}
        for i in tqdm(range(len(query_ids)), desc="Creating run"):
            query_sim = similarity[i]

            # Zip doc_ids and scores
            docs_scores = list(zip(self.doc_ids, query_sim))

            # Sort descending by similarity
            docs_scores.sort(key=lambda x: x[1], reverse=True)

            # Keep top_k if specified
            if top_k:
                docs_scores = docs_scores[:top_k]

            # Build the inner dict
            run[str(query_ids[i])] = {str(doc_id): float(score) for doc_id, score in docs_scores}

        print("Run created.")
        return run

    def embed_bge(self, texts, reuse_embeddings, path):
        """
        Embed the queries and documents using the BAAI embeddings models and compute the similarity between queries and documents.
        Calls the retrieve function.

        Args:
            docs (dict): Dictionary with document_id as key and text as value.
            queries (list): List of queries.
            top_k (int): Number of most similar documents to retrieve.

        Returns:
            dict: Dictionary with query as key and a list of tuples of (similarity, document text, doc_id) as value.
        """
        if not os.path.exists(path) or not reuse_embeddings:
            self.embeddings = self.model.encode(texts, batch_size=64, max_length=MAX_DOC_LEN)['dense_vecs']
            
            try:
                # save embeddings
                print("Saving embeddings...", end="")
                np.save(path, self.embeddings)
                print("Done.")
            except:
                print("Could not save embeddings.")
        else:
            # Load embeddings
            self.embeddings = np.load(path)

    def retrieve_top_k(self, query, top_k=5):
        """
        Embed the queries and documents using the BAAI embeddings models and compute the similarity between queries and documents.
        Calls the retrieve function.

        Args:
            model: BAAI embeddings model.
            docs (dict): Dictionary with document_id as key and text as value.
            queries (list): List of queries.
            top_k (int): Number of most similar documents to retrieve.

        Returns:
            dict: Dictionary with query_id as key and a dict of (doc_id, similarity) as value.
        """
        embedding_query = self.model.encode(query, max_length=MAX_QUERY_LEN)['dense_vecs']
        print(embedding_query)

        # Compute similarities
        similarity = self.compute_similarity(torch.tensor(embedding_query, dtype=torch.float32), torch.tensor(self.embeddings, dtype=torch.float32))
        
        run = self._retrieve(['qid_1'], similarity, top_k)['qid_1']

        top_doc_ids = [id for id, score in run.items()]

        top_docs = [self.doc_dict[id] for id in top_doc_ids]
        return top_docs