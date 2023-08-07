from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import numpy as np 
from tritonclient.utils import np_to_triton_dtype
import tritonhttpclient
from pydantic import BaseModel

class Message(BaseModel):
    message: str
    users: str 
    
app = FastAPI()
INPUT_NAMES = ['query']
OUTPUT_NAMES = ['e2e_answer']
MODEL_NAME = 'ensemble_model'
URL_TRITON = '0.0.0.0:8000'
MODEL_VERSION = '1'
VERBOSE = 1
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



triton_client = tritonhttpclient.InferenceServerClient(
        url=URL_TRITON,
        verbose=VERBOSE
    )

@app.get("/")
async def root():
    return {"message": "Hello wordl"}

@app.post('/answer')
def answer_question(question: Message):
    input_feature = np.array([bytes(question.message, 'utf8')], dtype=np.bytes_).reshape(1, 1)
    
    input0 = tritonhttpclient.InferInput(INPUT_NAMES[0], input_feature.shape, np_to_triton_dtype(input_feature.dtype))
    input0.set_data_from_numpy(input_feature)
    list_output = []
    for out in OUTPUT_NAMES:
        list_output.append(
            tritonhttpclient.InferRequestedOutput(
                out, binary_data=False
            )
        )
    response = triton_client.infer(model_name=MODEL_NAME, model_version=MODEL_VERSION, inputs=[input0], outputs=list_output)
    answer = response.as_numpy('e2e_answer')[0][0]
    if answer in ["", " "]:
        answer = "Hiện tại hệ thống chưa trả lời được câu hỏi của bạn!"
    return {'message': answer}