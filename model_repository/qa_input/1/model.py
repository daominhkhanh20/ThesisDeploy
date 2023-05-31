import triton_python_backend_utils as pb_utils
import numpy as np
import json
import numpy as np 
import os
from e2eqavn.processor import BM25Scoring
from e2eqavn.utils.io import load_json_data
from e2eqavn.utils.calculate import make_input_feature_qa
from e2eqavn.datasets import DataCollatorCustom
from torch.utils.dlpack import from_dlpack
from transformers import AutoTokenizer


class TritonPythonModel:
    def initialize(self, args):
        model_config = json.loads(args['model_config'])
        path_json_data = model_config['parameters']['path_corpus']['string_value']
        with open(path_json_data, 'r',encoding="utf-8") as file:
            data = json.load(file)
        self.list_document = [doc['context'] for doc in data]
        self.sbert_tokenizer = AutoTokenizer.from_pretrained('khanhbk20/vn-sentence-embedding')
        self.qa_tokenizer = AutoTokenizer.from_pretrained('khanhbk20/mrc_testing')
        self.input_names = ['sbert_index_selection', 'sbert_input_ids']
        self.output_names = ['input_ids', 'attention_mask', 'align_matrix']
        self.data_collator = DataCollatorCustom(tokenizer=self.qa_tokenizer, mode_triton=True)
    
    def decode(self, input_ids):
        return self.sbert_tokenizer.decode(input_ids, skip_special_tokens=True)
        
    def execute(self, requests):
        responses = []
        for request in requests:
            sbert_index = pb_utils.get_input_tensor_by_name(request, self.input_names[0]).as_numpy()[0]
            question = self.decode(pb_utils.get_input_tensor_by_name(request, self.input_names[0]).as_numpy()[0])
            input_features = self.data_collator(
                make_input_feature_qa(
                    questions=[question] * len(sbert_index),
                    documents=[self.list_document[idx] for idx in sbert_index],
                    tokenizer=self.qa_tokenizer
                )
            )
            
            output0 = pb_utils.Tensor('input_ids', np.array(input_features['input_ids']))
            output1 = pb_utils.Tensor('attention_mask', np.array(input_features['attention_mask']))
            output2 = pb_utils.Tensor('align_matrix', np.array(input_features['align_matrix']))
            
            responses.append(
                pb_utils.InferenceResponse(output_tensors=[output0, output1, output2])
            )             
        return responses

    def finalize(self):
        print('Cleaning up...')
                