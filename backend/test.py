import numpy as np
import json
import numpy as np 
import os
from e2eqavn.processor import BM25Scoring
from e2eqavn.retrieval import *
from e2eqavn.documents import Corpus
from e2eqavn.utils.io import load_json_data, load_yaml_file
from e2eqavn.keywords import *
from sentence_transformers import SentenceTransformer
import sys 
from e2eqavn.pipeline import E2EQuestionAnsweringPipeline
from unicodedata import normalize


corpus = Corpus.init_corpus(
    path_data='model/corpus.json'
)
bm25_retrieval = BM25Retrieval(corpus=corpus)
pipeline = E2EQuestionAnsweringPipeline(
    retrieval=[bm25_retrieval]
)
sentence = "NP-khó là gì?"
result = pipeline.run(
    queries=sentence,
    top_k_bm25=3,
)
for doc in result['documents'][0]:
    print(f"{doc.index} -- {doc.document_context}")
# bm25_scoring = BM25Scoring(corpus=corpus.list_document_context)
# list_document = corpus.list_document_context
# sentence = "NP-khó là gì?"
# mapping_idx_score = bm25_scoring.get_top_k(sentence, 15)
# print(mapping_idx_score)
# for idx in mapping_idx_score.keys():
#     print(f"{idx} -- {list_document[idx]}\n\n")