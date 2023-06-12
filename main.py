import tritonclient 
from tritonclient.utils import np_to_triton_dtype
import tritonhttpclient
import typing 
from fastapi import FastAPI
import numpy as np 

app = FastAPI()
INPUT_NAME = 'query'
OUTPUT_NAMES = ['e2e_answer']
MODEL_NAME = 'ensemble_model'
URL_TRITON = '0.0.0.0:8000'
MODEL_VERSION = '-1'
VERBOSE = 1

triton_client = tritonhttpclient.InferenceServerClient(
    url=URL_TRITON,
    verbose=VERBOSE
)

@app.post
def answer_question(sentence: str):
    input_feature = np.array(
        bytes(sentence, 'utf-8'), dtype=np.bytes_
    ).reshape(1, 1)
    
    input0 = tritonhttpclient.InferInput(INPUT_NAME, input0.shape, np_to_triton_dtype(input0.dtype))
    list_output = []
    for out in OUTPUT_NAMES:
        list_output.append(
            tritonhttpclient.InferRequestedOutput(
                out, binary_data=False
            )
        )
    response = triton_client.infer(model_name=MODEL_NAME, model_version=MODEL_VERSION, inputs=[input0], outputs=list_output)
    return {'answer': response.as_numpy('e2e_answer')}