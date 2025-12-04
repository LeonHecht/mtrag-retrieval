import dspy
import os
import retriever
import data

from config import API_KEY, STORAGE_DIR


lm = dspy.LM("cohere/command-a-03-2025", api_key=API_KEY)
dspy.configure(lm=lm)
corpus = "clapnq"


class Rewriter:
    def __init__(self) -> None:
        pass

