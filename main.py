from model import Retriever

retriever = Retriever(qtype="rewrite", corpus="clapnq")
top_docs = retriever.retrieve(query="What is Tesla?", top_k=5)

print(top_docs)