import numpy as np 
import sys 
import os 
import tritonclient
import tritonhttpclient
import struct
from unicodedata import normalize
from tritonclient.utils import np_to_triton_dtype



input_name = ['question']
output_name = ['bm25_index_selection', 'sbert_input_ids', 'sbert_attention_mask', 'token_type_ids', 'top_k_sbert']

def run_inference(sentence, model_name='bm25_retrieval', url='127.0.0.1:8000', model_version='1'):
    triton_client = tritonhttpclient.InferenceServerClient(
        url=url, verbose=False
    )
    input_feature = np.array([bytes(sentence, 'utf8')], dtype=np.bytes_).reshape(1, 1)
    
    input0 = tritonhttpclient.InferInput(input_name[0], input_feature.shape, np_to_triton_dtype(input_feature.dtype))
    input0.set_data_from_numpy(input_feature)
    list_output = []
    for out in output_name:
        list_output.append(
            tritonhttpclient.InferRequestedOutput(
                out, binary_data=False
            )
        )
    response = triton_client.infer(model_name=model_name, model_version=model_version, inputs=[input0], outputs=list_output)
    for out in output_name:
        print(response.as_numpy(out))
    
    
if __name__ == '__main__':
    import time 
    start_time = time.time()
    sentence = 'hello bạn hiền'
    run_inference(sentence)
    print(time.time() - start_time)