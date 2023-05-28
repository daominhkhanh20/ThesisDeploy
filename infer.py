import numpy as np 
import sys 
import os 
import tritonclient
import tritonhttpclient

input_name = ['input__0']
output_name = ['output__0']

def run_inference(sentence, model_name='bm25_retrieval', url='127.0.0.1:8000', model_version='1'):
    triton_client = tritonhttpclient.InferenceServerClient(
        url=url, verbose=False
    )
    
    input_feature = np.array([sentence.encode('utf-8')], dtype=np.object_)
    input0 = tritonhttpclient.InferInput(input_name[0], [1], 'BYTES')
    input0.set_data_from_nump(input_feature)
    output = tritonhttpclient.InferRequestedOutput(
        output_name[0], binary_data=False
    )
    response = triton_client.infer(model_name=model_name, model_version=model_version, inputs=[input0], outputs=[output])
    print(response)
    
    
if __name__ == '__main__':
    import time 
    start_time = time.time()
    run_inference('Phạm văn đồng')
    print(time.time() - start_time)