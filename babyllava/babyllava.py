'''
使用edgenext_samll做图像编码器
    torch.Size([8, 48, 128, 128])
    torch.Size([8, 96, 64, 64])
    torch.Size([8, 160, 32, 32])
    torch.Size([8, 304, 16, 16])
通道数比较固定（共608个通道），以通道数确定输入transformer的token数量，暂时考虑608/16=38。
在缩减通道数时使用通道注意力，特征图插值到16*16=256刚好为token维度
在构建代码的时候尽量把通道缩放比例，插值参数化，可以更改。
'''
import sys,os
sys.path.append(os.getcwd())
import torch
import torch.nn as nn
from typing import List, Literal, Optional, Tuple, TypedDict
from llama.model import ModelArgs, Transformer
from llama.tokenizer import Tokenizer
import config

class BabyLLaVa(nn.Module):
    @staticmethod
    def build(
        tokenizer_path: str,
        max_seq_len: int,
        max_batch_size: int,

    ) -> "BabyLLaVa":
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
        return BabyLLaVa(model, tokenizer)
    
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