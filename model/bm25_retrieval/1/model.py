import triton_python_backend_utils as pb_utils
import numpy as np
import json
import numpy as np 
import os
from e2eqavn.processor import BM25Scoring
from e2eqavn.utils.io import load_json_data

class TritonPythonModel:
    def initialize(self, args):
        model_config = json.loads(args['model_config'])
        path_json_data = model_config['parameters']['path_corpus']['string_value']
        self.top_k = int(model_config['parameters']['top_k']['string_value'])
        data = load_json_data(path_json_data)
        self.bm25_scoring = BM25Scoring(corpus=[document['context'] for document in data])
        output0_config = pb_utils.get_output_config_by_name(
            model_config, 'bm25_index_selection'
        )
        self.output0_dtype = pb_utils.triton_string_to_numpy(
            output0_config['data_type']
        )
        
    def execute(self, requests):
        responses = []
        for request in requests:
            in_0 = pb_utils.get_input_tensor_by_name(request, 'question')
            print(in_0)
            mapping_idx_score = self.bm25_scoring.get_top_k(in_0, self.top_k)
            print(mapping_idx_score)
            output0 = pb_utils.Tensor('bm25_index_selection', np.array(mapping_idx_score.keys()).astype(self.output0_dtype))
            responses.append(
                pb_utils.InferenceResponse(output_tensors=[output0])
            )             
        return responses

    def finalize(self):
        print('Cleaning up...')
                