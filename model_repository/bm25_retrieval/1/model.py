import triton_python_backend_utils as pb_utils
import numpy as np 
import json


class TritonPythonModel:
    def initialize(self, args):
       model_config = json.loads(args['model_config'])
       print(model_config)
        