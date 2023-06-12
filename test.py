import json
import numpy as np 
import os
from e2eqavn.documents import Corpus
from e2eqavn.processor import BM25Scoring
from e2eqavn.utils.io import load_json_data, load_yaml_file
from e2eqavn.utils.calculate import make_input_feature_qa
from e2eqavn.datasets import DataCollatorCustom
from transformers import AutoTokenizer
from e2eqavn.keywords import *
import sys 

config_pipeline = load_yaml_file('model/train_qa.yaml')
corpus = Corpus.init_corpus(
    path_data=config_pipeline[DATA][PATH_TRAIN],
    **config_pipeline.get(CONFIG_DATA, {})
)