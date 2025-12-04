from retriever import Retriever
from data import Data
import os
from config import STORAGE_DIR


def get_bge_m3_model(checkpoint):
    """ Load BAAI embeddings model."""
    from FlagEmbedding import BGEM3FlagModel
    print("Loading BAAI embeddings model from checkpoint:", checkpoint)
    # model = BGEM3FlagModel(checkpoint, use_fp16=True) # Setting use_fp16 to True speeds up computation with a slight performance degradation
    model = BGEM3FlagModel(checkpoint) # Setting use_fp16 to True speeds up computation with a slight performance degradation
    return model


qtype = "rewrite"
corpus = "clapnq"

data = Data(qtype, corpus)

model = get_bge_m3_model('BAAI/bge-m3')
emb_path = os.path.join(STORAGE_DIR, f'mtrag/data/corpus/embeddings/corpus_{corpus}_bge-m3-512.npy')

retriever = Retriever(model, doc_ids=data.doc_ids, docs=data.docs, emb_path=emb_path)
top_docs = retriever.retrieve_top_k(query="What is Tesla?", top_k=5)

print(top_docs)