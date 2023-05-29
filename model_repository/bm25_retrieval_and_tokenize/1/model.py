import triton_python_backend_utils as pb_utils
import numpy as np
import json
import numpy as np 
import os
from e2eqavn.processor import BM25Scoring
from e2eqavn.utils.io import load_json_data
from torch.utils.dlpack import from_dlpack
from sentence_transformers import SentenceTransformer



class TritonPythonModel:
    def initialize(self, args):
        model_config = json.loads(args['model_config'])
        path_json_data = model_config['parameters']['path_corpus']['string_value']
        self.top_k = int(model_config['parameters']['top_k']['string_value'])
        with open(path_json_data, 'r',encoding="utf-8") as file:
            data = json.load(file)
        self.bm25_scoring = BM25Scoring(corpus=[document['context'] for document in data])
        self.output0_dtype = pb_utils.triton_string_to_numpy(
            pb_utils.get_output_config_by_name(
                model_config, 'bm25_index_selection'
            )['data_type']
        )
        self.output1_dtype = pb_utils.triton_string_to_numpy(
            pb_utils.get_output_config_by_name(
                model_config, 'sbert_input_ids'
            )['data_type']
        )
        self.output2_dtype = pb_utils.triton_string_to_numpy(
            pb_utils.get_output_config_by_name(
                model_config, 'sbert_attention_mask'
            )['data_type']
        )
        self.output3_dtype = pb_utils.triton_string_to_numpy(
            pb_utils.get_output_config_by_name(
                model_config, 'token_type_ids'
            )['data_type']
        )
        self.encoder = SentenceTransformer('khanhbk20/vn-sentence-embedding', device='cpu')
        
    def execute(self, requests):
        responses = []
        for request in requests:
            in_0 = pb_utils.get_input_tensor_by_name(request, 'question')
            print(in_0.as_numpy().astype(np.bytes_))
            sentence = in_0.as_numpy().astype(np.bytes_)[0][0].decode('utf-8')
            print(sentence)
            mapping_idx_score = self.bm25_scoring.get_top_k(sentence, self.top_k)
            output_tokenizer = self.encoder.tokenize(sentence)
            output0 = pb_utils.Tensor('bm25_index_selection', np.array(list(mapping_idx_score.keys())).astype(self.output0_dtype))
            output1 = pb_utils.Tensor('sbert_input_ids', np.array(output_tokenizer['input_ids']).astype(self.output1_dtype))
            output2 = pb_utils.Tensor('sbert_attention_mask', np.array(output_tokenizer['attention_mask']).astype(self.output2_dtype))
            output3 = pb_utils.Tensor('token_type_ids', np.array(output_tokenizer['token_type_ids']).astype(self.output3_dtype))
            responses.append(
                pb_utils.InferenceResponse(output_tensors=[output0, output1, output2, output3])
            )             
        return responses

    def finalize(self):
        print('Cleaning up...')
                