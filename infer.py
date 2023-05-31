import numpy as np 
import sys 
import os 
import tritonclient
import tritonhttpclient
import struct
from unicodedata import normalize
from tritonclient.utils import np_to_triton_dtype



input_name = ['query']
output_name = ['final_retrieval_stage_selection', 'retrieval_input_ids']

def run_inference(sentence, model_name='ensemble_model', url='127.0.0.1:8000', model_version='1'):
    triton_client = tritonhttpclient.InferenceServerClient(
        url=url, verbose=1
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
    
    print(response.as_numpy('query'))
    
    
if __name__ == '__main__':
    import time 
    start_time = time.time()
    sentence = 'Paris nằm ở điểm gặp nhau của các hành trình'
    run_inference(sentence)
    print(time.time() - start_time)