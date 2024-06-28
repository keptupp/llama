import sys,os
sys.path.append(os.getcwd())
import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
import pandas as pd
from llama.tokenizer import Tokenizer
import config
from tqdm import tqdm

def read_REDGPTJSON(text_path,tokenizer):
    data=pd.read_json(text_path,lines=True)

    total_chat_token=[]
    for row in data.itertuples():
        text_list=getattr(row,'dialogue').replace("Assistant","助手").replace("Human","人类").split("\n\n人类")

        first=True
        chat_token=tokenizer.encode("资料："+getattr(row,'reference'),bos=True,eos=True)
        for one in text_list:
            if first:
                one="\n\n"+one
                one_token=tokenizer.encode(one,bos=False,eos=True)#第一段对话
                first=False
            else:
                one="\n\n人类"+one 
                one_token=tokenizer.encode(one,bos=False,eos=True)#第二段对话，补全人类

            chat_token=chat_token+one_token

        total_chat_token.append(chat_token)
    print("REDGPT数据集",len(total_chat_token)/1e6,"M")
    return total_chat_token

def read_NaturalConvJSON(text_path,tokenizer):
    data=pd.read_json(text_path)
    total_chat_token=[]
    for row in data.itertuples():
        # print(row)
        text_list=getattr(row,'content')

        role="人类"
        first=True
        chat_token=tokenizer.encode("资料：",bos=True,eos=True)
        for one in text_list:
            if role=="人类":
                one="\n\n人类: "+one 
                one_token=tokenizer.encode(one,bos=False,eos=False)#第二段对话，补全人类
                role="助手"
            elif role=="助手":
                one="\n\n助手: "+one 
                one_token=tokenizer.encode(one,bos=False,eos=True)#第二段对话，补全人类
                role="人类"

            chat_token=chat_token+one_token

        total_chat_token.append(chat_token)
    print("NaturalConv数据集",len(total_chat_token)/1e6,"M")
    return total_chat_token

def read_pCLUEJSON(text_path,tokenizer):
    total_chat_token=[]

    def read_one_data(text_path,tokenizer):
        data=pd.read_json(text_path,lines=True)
        total_chat_token=[]
        for row in data.itertuples():
            question=getattr(row,'input').replace("\n答案：","")
            answer=getattr(row,'target')
            one_token=tokenizer.encode("资料: ",bos=True,eos=True)
            one_token+=tokenizer.encode("\n\n人类: "+question,bos=False,eos=False)
            one_token+=tokenizer.encode("\n\n助手: "+answer,bos=False,eos=True)

            total_chat_token.append(one_token)
        return total_chat_token
    
    for one_path in text_path:
        total_chat_token+=read_one_data(one_path,tokenizer)
    print("pCLUE数据集",len(total_chat_token)/1e6,"M")
    return total_chat_token

def read_BelleGroup(text_path,tokenizer):
    data=pd.read_json(text_path,lines=True)
    total_chat_token=[]
    for row in data.itertuples():
        chat_token=tokenizer.encode("资料: ",bos=True,eos=True)
        chat_token+=tokenizer.encode("\n\n人类: "+getattr(row,'instruction'),bos=False,eos=False)
        chat_token+=tokenizer.encode("\n\n助手: "+getattr(row,'output'),bos=False,eos=True)

        total_chat_token.append(chat_token)
    print("BelleGroup数据集",len(total_chat_token)/1e6,"M")
    return total_chat_token

def read_SFT_json(text_path,tokenizer):
    data=pd.read_json(text_path,lines=True,chunksize=1e5)
    total_chat_token=[]
    nums=0
    for one_data in data:
        # for row in one_data.itertuples():
        #     history=getattr(row,'history')
        #     if(len(history)>=1):
        #         history="".join(history[0])
        #     else:
        #         history=""
        #     chat_token=tokenizer.encode("资料: "+history,bos=True,eos=True)
        #     chat_token+=tokenizer.encode("\n\n人类: "+getattr(row,'input'),bos=False,eos=False)
        #     chat_token+=tokenizer.encode("\n\n助手: "+getattr(row,'output'),bos=False,eos=True)

        #     total_chat_token.append(chat_token)
        print("1111")
        nums+=1
    
    print(nums)
    print("SFT数据集",len(total_chat_token)/1e6,"M")
    return total_chat_token

#聊天，每次助手对完话均有一个结束符号3
class ChatDataset(Dataset):
    def __init__(self,text_path,tokenizer_path,min_len,max_len,):
       
        self.data_section=[]
        self.tokenizer=Tokenizer(tokenizer_path)
        self.max_len=max_len

        self.data_section+=read_REDGPTJSON(text_path["redgpt"],self.tokenizer)
        
        self.data_section+=read_NaturalConvJSON(text_path["naturalconv"],self.tokenizer)

        # self.data_section+=read_pCLUEJSON(text_path["pclue"],self.tokenizer)
        

    def __getitem__(self, index):
        chat_token=self.data_section[index]
        if(len(chat_token)>self.max_len):
            chat_token=(chat_token[:self.max_len-1]+[3])
        token=torch.tensor(chat_token, dtype=torch.long, device="cuda")
        return token[:-1].detach().clone(),token[1:].detach().clone()

    def __len__(self):
        return len(self.data_section)

if __name__=="__main__":



    tokenizer=Tokenizer("weight/tokenizer_chinese.model")


    # read_REDGPTJSON(r"D:\work\Datasets\RedGPT-Dataset-V1-CN\RedGPT-Dataset-V1-CN.json",tokenizer)
    
    # token=read_NaturalConvJSON(r"D:\work\Datasets\NaturalConv_Release_20210318\NaturalConv.json",tokenizer=tokenizer)
    # for one in token:
    #     print(tokenizer.decode(one))

    # text_path=[
    #     r"D:\work\Datasets\pCLUE-main\datasets\pCLUE_train_1.json",
    #     r"D:\work\Datasets\pCLUE-main\datasets\pCLUE_train_2.json"
    # ]
    # token=read_pCLUEJSON(text_path,tokenizer)
    # print(tokenizer.decode(token[100]))

    # text_path=r"D:\work\Datasets\Belle_open_source_0.5M.json"
    # read_BelleGroup(text_path,tokenizer)

    text_path=r"D:\work\Datasets\sft_data_zh.jsonl"
    read_SFT_json(text_path,tokenizer)