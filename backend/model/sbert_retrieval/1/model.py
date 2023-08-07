import triton_python_backend_utils as pb_utils
import numpy as np
import json
import numpy as np 
import os
from e2eqavn.pipeline import E2EQuestionAnsweringPipeline
from e2eqavn.retrieval import *
from e2eqavn.documents import Corpus
from e2eqavn.utils.io import load_json_data, load_yaml_file
from e2eqavn.keywords import *
from sentence_transformers import SentenceTransformer
import sys 
from unicodedata import normalize
sys.stdout.reconfigure(encoding="utf-8")

class TritonPythonModel:
    def initialize(self, args):
        model_config = json.loads(args['model_config'])
        corpus = Corpus.init_corpus(
            path_data='model/corpus.json'
        )
        self.logger = pb_utils.Logger
        self.sbert_retrieval = SBertRetrieval.from_pretrained(model_name_or_path='khanhbk20/vn-sentence-embedding')
        self.sbert_retrieval.update_embedding(corpus=corpus)
        self.output0_dtype = pb_utils.triton_string_to_numpy(
            pb_utils.get_output_config_by_name(
                model_config, 'sbert_index_selection'
            )['data_type']
        )
    
    def get_result(self, queries, index_selection):
        scores, top_k_indexs = self.sbert_retrieval.query_by_embedding(queries, top_k=2,
                                                       index_selection=index_selection)
        scores = scores.cpu().numpy()
        top_k_indexs = top_k_indexs.cpu().numpy()
        return top_k_indexs
        
        
    def execute(self, requests):
        responses = []
        for request in requests:
            in_0 = pb_utils.get_input_tensor_by_name(request, 'question1')
            sentence = in_0.as_numpy().astype(np.bytes_)[0][0].decode('utf-8')
            sentence = normalize("NFC", sentence).lower()
            index_selection = pb_utils.get_input_tensor_by_name(request, 'bm25_index_selection').as_numpy()
            output0 = self.get_result([sentence], index_selection=index_selection)
            self.logger.log_info(f"Sbert selection: {output0}")
            responses.append(
                pb_utils.InferenceResponse(output_tensors=[output0])
            )             
        return responses

    def finalize(self):
        print('Cleaning up...')