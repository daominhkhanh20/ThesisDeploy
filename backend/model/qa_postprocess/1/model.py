import sys
import torch 
import numpy as np 
import triton_python_backend_utils as pb_utils
import json 
from transformers import AutoTokenizer

class TritonPythonModel:
    def initialize(self, args):
        model_config = json.loads(args['model_config'])
        tokenizer_name = model_config['parameters']['tokenizer_name']['string_value']
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.input_names = ['e2e_start_logits', 'e2e_end_logits', 'e2e_input_ids', 'e2e_align_matrix']
        self.output_names = ['answer']
        self.output0_dtype = pb_utils.triton_string_to_numpy(
            pb_utils.get_output_config_by_name(
                model_config, self.output_names[0]
            )['data_type']
        )
        self.logger = pb_utils.Logger
        
    def get_best_choice(self, start_logits, end_logits):
        start_logits = torch.softmax(start_logits, dim=-1)
        end_logits = torch.softmax(end_logits, dim=-1)
        
        start_scores, start_idxs = torch.max(start_logits, dim=-1)
        end_scores, end_idxs = torch.max(end_logits, dim=-1)
        total_scores = torch.mul(start_scores.reshape(-1), end_scores.reshape(-1))
        self.logger.log_info(f"Start score: {start_scores} \n" + 
                             f"End score: {end_scores}\n" + 
                             f"Final score: {total_scores}")
        best_index = torch.argmax(total_scores)
        return best_index, start_idxs[best_index].item(), end_idxs[best_index].item()
    
    def parser_answer(self, input_ids, words_length, start_location, end_location):
        answer_start_idx = sum(words_length[: start_location])
        answer_end_idx = sum(words_length[: end_location + 1])
        return self.tokenizer.convert_tokens_to_string(
                    self.tokenizer.convert_ids_to_tokens(input_ids[answer_start_idx:answer_end_idx])
                )
        
    
    def execute(self, requests):
        responess = []
        for request in requests:
            start_logits = torch.tensor(pb_utils.get_input_tensor_by_name(request, self.input_names[0]).as_numpy())
            end_logits = torch.tensor(pb_utils.get_input_tensor_by_name(request, self.input_names[1]).as_numpy())
            input_ids = torch.tensor(pb_utils.get_input_tensor_by_name(request, self.input_names[2]).as_numpy())
            align_matrix = torch.tensor(pb_utils.get_input_tensor_by_name(request, self.input_names[3]).as_numpy())
            words_length = torch.sum(align_matrix, dim=-1).to(torch.int32)
            best_choice, start_location, end_location = self.get_best_choice(start_logits, end_logits)
            self.logger.log_info(f"Start location: {start_location} \n" + 
                             f"End location: {end_location}\n"
                             )
            answer = self.parser_answer(
                input_ids=input_ids[best_choice, :].reshape(-1),
                words_length=words_length[best_choice, :].reshape(-1),
                start_location=start_location,
                end_location=end_location
            )
            output0 = pb_utils.Tensor(self.output_names[0], np.array([answer.encode('utf-8')], dtype=np.bytes_).reshape(1, -1))
            responess.append(
                pb_utils.InferenceResponse(output_tensors=[output0])
            )
        return responess
            
            
        