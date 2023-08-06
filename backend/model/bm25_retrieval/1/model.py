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
        path_json_data = model_config['parameters']['path_corpus']['string_value']
        path_config = model_config['parameters']['path_config']['string_value']
        
        self.top_k_bm25 = int(model_config['parameters']['top_k_bm25']['string_value'])
        self.top_k_sbert = int(model_config['parameters']['top_k_sbert']['string_value'])
        config_pipeline = load_yaml_file(path_config)
        corpus = Corpus.init_corpus(
            path_data=config_pipeline[DATA][PATH_TRAIN],
            **config_pipeline.get(CONFIG_DATA, {})
        )
        bm25_retrieval = BM25Retrieval(corpus=corpus)
        self.pipeline = E2EQuestionAnsweringPipeline(
            retrieval=[bm25_retrieval]
        )
        self.output0_dtype = pb_utils.triton_string_to_numpy(
            pb_utils.get_output_config_by_name(
                model_config, 'index_selection'
            )['data_type']
        )
        self.output1_dtype = pb_utils.triton_string_to_numpy(
            pb_utils.get_output_config_by_name(
                model_config, 'sentence_bert_input_ids'
            )['data_type']
        )
        self.output2_dtype = pb_utils.triton_string_to_numpy(
            pb_utils.get_output_config_by_name(
                model_config, 'sentence_bert_attention_mask'
            )['data_type']
        )
        self.output3_dtype = pb_utils.triton_string_to_numpy(
            pb_utils.get_output_config_by_name(
                model_config, 'sentence_bert_token_type_ids'
            )['data_type']
        )
        self.output4_dtype = pb_utils.triton_string_to_numpy(
            pb_utils.get_output_config_by_name(
                model_config, 'top_k'
            )['data_type']
        )
        self.output5_dtype = pb_utils.triton_string_to_numpy(
            pb_utils.get_output_config_by_name(
                model_config, 'bm25_scoring'
            )['data_type']
        )
        self.logger = pb_utils.Logger
        self.encoder = SentenceTransformer('khanhbk20/vn-sentence-embedding', device='cpu')
    
    def get_result(self, sentence):
        result = self.pipeline.run(
            queries=sentence,
            top_k_bm25=self.top_k_bm25
        )
        mapping_idx_score = {}
        for doc in result['documents'][0]:
            mapping_idx_score[doc.index] = doc.bm25_score
        return mapping_idx_score
        
    def execute(self, requests):
        responses = []
        for request in requests:
            in_0 = pb_utils.get_input_tensor_by_name(request, 'question')
            sentence = in_0.as_numpy().astype(np.bytes_)[0][0].decode('utf-8')
            sentence = normalize("NFC", sentence).lower()
            mapping_idx_score = self.get_result(sentence)
            self.logger.log_info(f"BM25 mapping score: {mapping_idx_score}")
            output_tokenizer = self.encoder.tokenize([sentence])
            output0 = pb_utils.Tensor('index_selection', np.array(list(mapping_idx_score.keys())).astype(self.output0_dtype).reshape(1, -1))
            output1 = pb_utils.Tensor('sentence_bert_input_ids', np.array(output_tokenizer['input_ids']).astype(self.output1_dtype))
            output2 = pb_utils.Tensor('sentence_bert_attention_mask', np.array(output_tokenizer['attention_mask']).astype(self.output2_dtype))
            output3 = pb_utils.Tensor('sentence_bert_token_type_ids', np.array(output_tokenizer['token_type_ids']).astype(self.output3_dtype))
            output4 = pb_utils.Tensor('top_k', np.array([self.top_k_sbert]).astype(self.output4_dtype).reshape(1, -1))
            print(f"Top k sbert: {self.top_k_sbert} -- {self.top_k_bm25}")
            output5 = pb_utils.Tensor('bm25_scoring', np.array(list(mapping_idx_score.values())).astype(self.output5_dtype).reshape(1, -1))
            responses.append(
                pb_utils.InferenceResponse(output_tensors=[output0, output1, output2, output3, output4, output5])
            )             
        return responses

    def finalize(self):
        print('Cleaning up...')
                