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
from llama.model import ModelArgs, Transformer_Vision
from llama.tokenizer import Tokenizer
from edgenext.edgenext import EdgeNeXt
from edgenext.model import edgenext_small
from llama.generation import sample_top_p
import config
import torch.nn.functional as F


class SEBlock(nn.Module):
    def __init__(self, channel,reduction=1):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class Project_W(nn.Module):
    #暂时不编写配置文件，直接使用默认的
    def __init__(self,channels=[48,96,160,304],reduction=16):
        super().__init__()
        #初步想法是将特征图缩小到16*16，通道数缩小16倍，然后reshape
        self.channels=channels
        self.seblocks=nn.ModuleList([SEBlock(c) for c in self.channels])

        #通道减少应该放在减小分辨率前，可以用更多的特征来选择适合的通道
        self.reduce_channels=nn.ModuleList([nn.Conv2d(in_channels=c,out_channels=c//reduction,kernel_size=3,padding=1) for c in self.channels])
    
    def forward(self,features):
        
        #通道注意力提取重要特征
        features=[se(feat) for se,feat in zip(self.seblocks,features)]
        #卷积网络减少通道数量
        features=[reduce_cnn(feat) for reduce_cnn,feat in zip(self.reduce_channels,features)]
        #双线性插值减小特征图尺寸，统一到16*16
        features=[F.interpolate(feat, size=(16, 16), mode='bilinear') for feat in features]

        b,c,h,w=features[0].shape

        image_tokens=features[0]
        for feat in features[1:]:
            image_tokens=torch.cat([image_tokens,feat],dim=1)
        
        b,c,h,w=image_tokens.shape
        image_tokens=image_tokens.reshape(b,c,h*w)
        return image_tokens
        


class BabyLLaVa(nn.Module):
    @staticmethod
    def build(
        tokenizer_path: str,
    ) -> "BabyLLaVa":
        model_args: ModelArgs = ModelArgs(
            max_seq_len=2048,
            max_batch_size=32,
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
        
        model = Transformer_Vision(model_args)

        image_encoder=edgenext_small()
        # image_encoder.load_state_dict(torch.load("weight/edgenext_small.pth")["model"],strict=False)#去掉头的权重
        print("模型配置参数",model_args)
        babyllava=BabyLLaVa(model, tokenizer ,image_encoder)
        num_params = sum(p.numel() for p in babyllava.parameters() if p.requires_grad)
        print(f'模型参数量: {num_params/1e6}M')
        return babyllava.to(config.device)
    
    def __init__(self, model: Transformer_Vision, tokenizer: Tokenizer, image_encoder: EdgeNeXt):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.image_encoder=image_encoder
        self.project_w=Project_W()

    def forward(
        self,
        prompt_tokens: List[List[int]],
        images,
        prev_pos=0
    ):
        image_features=self.image_encoder(images)

        # for one in image_features:
        #     print(one.shape)

        image_tokens=self.project_w(image_features)

        logits = self.model(prompt_tokens,image_tokens ,prev_pos)
        return logits
    def inference(self,token,image,prev_pos=0,max_length=256,top_p=0):
        pre_text=[]
        image_features=self.image_encoder(image)
        image_tokens=self.project_w(image_features)
        #目前这个方法还只适用于batchsize=1
        for i in range(max_length):
            logits = self.model(token,image_tokens,prev_pos)

            temperature=1
            pre_tokens = torch.softmax(logits[:, -1] / temperature, dim=-1)

            if(top_p>0):
                pre_tokens = sample_top_p(pre_tokens, top_p)
            else:
                pre_tokens=torch.argmax(pre_tokens,dim=-1).unsqueeze(0)

            last_token=pre_tokens[:,-1].item()

            pre_text.append(last_token)

            if(last_token==self.tokenizer.eos_id):
                break

            token=torch.cat([token,pre_tokens[:,-1].unsqueeze(0)],dim=1)
        return [pre_text]
    

if __name__=="__main__":
    babyllava=BabyLLaVa.build(r"weight\tokenizer.model")

    tokens=torch.rand(8,50).long().cuda()
    images=torch.rand(8,3,512,512).cuda()

    babyllava(tokens,images)