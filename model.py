import dspy
import os
import retriever
import data

from config import API_KEY, STORAGE_DIR


lm = dspy.LM("cohere/command-a-03-2025", api_key=API_KEY)
dspy.configure(lm=lm)
corpus = "clapnq"


def get_bge_m3_model(checkpoint):
    """ Load BAAI embeddings model."""
    from FlagEmbedding import BGEM3FlagModel
    print("Loading BAAI embeddings model from checkpoint:", checkpoint)
    # model = BGEM3FlagModel(checkpoint, use_fp16=True) # Setting use_fp16 to True speeds up computation with a slight performance degradation
    model = BGEM3FlagModel(checkpoint) # Setting use_fp16 to True speeds up computation with a slight performance degradation
    return model


class Retriever:
    def __init__(self, qtype, corpus) -> None:
        self.data = data.Data(qtype, corpus)
        self.model = get_bge_m3_model('BAAI/bge-m3')
        self.emb_path = os.path.join(STORAGE_DIR, f'mtrag/data/corpus/embeddings/corpus_{corpus}_bge-m3-512.npy')

    def retrieve(self, query, top_k):
        top_doc_ids = retriever.retrieve_top_k(self.model,
                                            query,
                                            self.data.doc_ids,
                                            path=self.emb_path,
                                            top_k=top_k)
        top_docs = [self.data.doc_dict[id] for id in top_doc_ids]
        return top_docs


class Rewriter:
    def __init__(self) -> None:
        pass

