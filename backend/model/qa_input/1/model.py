import triton_python_backend_utils as pb_utils
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
sys.stdout.reconfigure(encoding="utf-8")

class TritonPythonModel:
    def initialize(self, args):
        model_config = json.loads(args['model_config'])
        path_json_data = model_config['parameters']['path_corpus']['string_value']
        path_config = model_config['parameters']['path_config']['string_value']
        tokenizer_name = model_config['parameters']['tokenizer_name']['string_value']
        config_pipeline = load_yaml_file(path_config)
        corpus = Corpus.init_corpus(
            path_data=config_pipeline[DATA][PATH_TRAIN],
            **config_pipeline.get(CONFIG_DATA, {})
        )
        self.list_documents = [doc.document_context for doc in corpus.list_document]
        self.qa_tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.input_names = ['qa_document_indexs', 'query_user', 'score']
        self.output_names = ['input_ids', 'attention_mask', 'align_matrix', 'score_reshape']
        self.data_collator = DataCollatorCustom(tokenizer=self.qa_tokenizer, mode_triton=True)
        self.output0_dtype = pb_utils.triton_string_to_numpy(
            pb_utils.get_output_config_by_name(
                model_config, self.output_names[0]
            )['data_type']
        )
        self.output1_dtype = pb_utils.triton_string_to_numpy(
            pb_utils.get_output_config_by_name(
                model_config, self.output_names[1]
            )['data_type']
        )
        self.output2_dtype = pb_utils.triton_string_to_numpy(
            pb_utils.get_output_config_by_name(
                model_config, self.output_names[2]
            )['data_type']
        )
        self.logger = pb_utils.Logger
    
    def decode(self, input_ids):
        return self.sbert_tokenizer.decode(input_ids, skip_special_tokens=True)
    
        
    def execute(self, requests):
        responses = []
        for request in requests:
            index_selections = pb_utils.get_input_tensor_by_name(request, self.input_names[0]).as_numpy()[0]
            self.logger.log_info(f"Sbert index selection: {index_selections}")
            for idx in index_selections:
                self.logger.log_info(f"{idx} -- {self.list_documents[idx]}")
            question = pb_utils.get_input_tensor_by_name(request, self.input_names[1]).as_numpy().astype(np.bytes_)[0][0].decode('utf-8')
            input_feature_raw = make_input_feature_qa(
                questions=[question] * len(index_selections),
                documents=[self.list_documents[idx] for idx in index_selections],
                tokenizer=self.qa_tokenizer,
                max_length=400
            )
            input_features = self.data_collator(input_feature_raw)
            for key, value in input_features.items():
                self.logger.log_info(f"{key} -- {value.size()}")
            output0 = pb_utils.Tensor(self.output_names[0], np.array(input_features[self.output_names[0]]))
            output1 = pb_utils.Tensor(self.output_names[1], np.array(input_features[self.output_names[1]]))
            output2 = pb_utils.Tensor(self.output_names[2], np.array(input_features[self.output_names[2]]))
            
            responses.append(
                pb_utils.InferenceResponse(output_tensors=[output0, output1, output2])
            )
        return responses

    def finalize(self):
        print('Cleaning up...')
                