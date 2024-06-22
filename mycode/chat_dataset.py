import sys,os
sys.path.append(os.getcwd())
import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
import pandas as pd
from llama.tokenizer import Tokenizer
import config
from tqdm import tqdm

#聊天，每次助手对完话均有一个结束符号3
class ChatDataset(Dataset):
    def __init__(self,text_path,tokenizer_path,min_len,max_len):
        data=pd.read_json(text_path,lines=True)
        self.data_section=[]
        self.reference_list=[]
        for row in data.itertuples():
            # print(row)
            #在每个人类前
            text_list=getattr(row,'dialogue').replace("Assistant","助手").replace("Human","人类").split("\n\n人类")
            self.reference_list.append("资料："+getattr(row,'reference'))

            self.data_section.append(text_list)

        self.tokenizer=Tokenizer(tokenizer_path)

    def __getitem__(self, index):
        chat_total=self.data_section[index]

        chat_token=self.tokenizer.encode(self.reference_list[index],bos=True,eos=True)

        # chat_token=[]
        first=True
        for one in chat_total:
            if first:
                one_token=self.tokenizer.encode(one,bos=False,eos=True)#第一段对话
                first=False
            else:
                one="\n\n人类"+one 
                one_token=self.tokenizer.encode(one,bos=False,eos=True)#第二段对话，补全人类

            chat_token=chat_token+one_token
        
        token=torch.tensor(chat_token, dtype=torch.long, device="cuda")
        return token[:-1].detach().clone(),token[1:].detach().clone()

    def __len__(self):
        return len(self.data_section)
    

if __name__=="__main__":
    pre_dataset=ChatDataset(r"D:\work\Datasets\RedGPT-Dataset-V1-CN\RedGPT-Dataset-V1-CN.json",r"weight/tokenizer.model",min_len=32,max_len=2048)
    pre_dataloader=DataLoader(pre_dataset,batch_size=1,shuffle=True)

    tokenizer=Tokenizer("weight/tokenizer_chinese.model")
    print(len(pre_dataset))
    mean_token=0
    for token,last_token in tqdm(pre_dataloader):
        # print(tokenizer.decode(token.cpu().tolist()[0]))
        mean_token+=(token.shape[1]/50000)

    print("平均语料长度",mean_token)