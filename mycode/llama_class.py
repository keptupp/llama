# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import json
import os
import sys
sys.path.append(os.getcwd())
import time
from pathlib import Path
from typing import List, Literal, Optional, Tuple, TypedDict

import torch
import torch.nn.functional as F
from fairscale.nn.model_parallel.initialize import (
    get_model_parallel_rank,
    initialize_model_parallel,
    model_parallel_is_initialized,
)


from llama.model import ModelArgs, Transformer
from llama.tokenizer import Tokenizer
import config
import torch.nn as nn

Role = Literal["system", "user", "assistant"]


class Message(TypedDict):
    role: Role
    content: str


class CompletionPrediction(TypedDict, total=False):
    generation: str
    tokens: List[str]  # not required
    logprobs: List[float]  # not required


class ChatPrediction(TypedDict, total=False):
    generation: Message
    tokens: List[str]  # not required
    logprobs: List[float]  # not required


Dialog = List[Message]

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

SPECIAL_TAGS = [B_INST, E_INST, "<<SYS>>", "<</SYS>>"]
UNSAFE_ERROR = "Error: special tags are not allowed as part of the prompt."


class Llama(nn.Module):
    @staticmethod
    def build(
        tokenizer_path: str,
        max_seq_len: int,
        max_batch_size: int,
        model_parallel_size: Optional[int] = None,
        seed: int = 1,
    ) -> "Llama":
        """
        Build a Llama instance by initializing and loading a pre-trained model.

        Args:
            ckpt_dir (str): Path to the directory containing checkpoint files.
            tokenizer_path (str): Path to the tokenizer file.
            max_seq_len (int): Maximum sequence length for input text.
            max_batch_size (int): Maximum batch size for inference.
            model_parallel_size (Optional[int], optional): Number of model parallel processes.
                If not provided, it's determined from the environment. Defaults to None.

        Returns:
            Llama: An instance of the Llama class with the loaded model and tokenizer.

        Raises:
            AssertionError: If there are no checkpoint files in the specified directory,
                or if the model parallel size does not match the number of checkpoint files.

        Note:
            This method initializes the distributed process group, sets the device to CUDA,
            and loads the pre-trained model and tokenizer.

        """
       
        model_args: ModelArgs = ModelArgs(
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
        )
        tokenizer = Tokenizer(model_path=tokenizer_path)
        model_args.vocab_size = tokenizer.n_words
        print("词袋大小",model_args.vocab_size)

        #尝试自定义
        model_args.dim=256
        model_args.n_layers=12
        model_args.n_heads=8
        

        # torch.set_default_tensor_type(torch.cuda.HalfTensor)
        # torch.set_default_tensor_type(torch.cuda.Tensor)
        
        model = Transformer(model_args).to(config.device)

        print("模型配置参数",model_args)
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f'模型参数量: {num_params/1e6}M')
        return Llama(model, tokenizer)

    def __init__(self, model: Transformer, tokenizer: Tokenizer):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer

    
    def forward(
        self,
        prompt_tokens: List[List[int]],
        prev_pos=0
    ):

        logits = self.model(prompt_tokens, prev_pos)
        
        return logits
    
    def inference(
        self,
        prompt_tokens: List[List[int]],
        prev_pos=0,
        max_length=256
    ):  
        pre_text=[]
        #目前这个方法还只适用于batchsize=1
        for i in range(max_length):
            logits = self.model(prompt_tokens, prev_pos)
            pre_tokens=torch.argmax(torch.softmax(logits,dim=-1),dim=-1)

            last_token=pre_tokens[:,-1].item()

            pre_text.append(last_token)

            if(last_token==self.tokenizer.eos_id):
                break

            prompt_tokens=torch.cat([prompt_tokens,pre_tokens[:,-1].unsqueeze(0)],dim=1)

            # print(prompt_tokens.shape)
    
        return [pre_text]
