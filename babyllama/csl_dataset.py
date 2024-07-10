import sys,os
sys.path.append(os.getcwd())
import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
import pandas as pd
from llama.tokenizer import Tokenizer
import config
from tqdm import tqdm
import json
import random

questions=[
    "生成一个能够概括以下文章的句子。",
    "请用几句话概括以下文章。",
    "将下面的文章转换成一个简短的概述。",
    "请提取出以下文章的关键信息并生成一个标题。",
    "请将以下文章总结为一段简洁的摘要。",
    "请将下面这篇文章的要点总结成一段简洁的摘要。",
    "请将下面的文章总结为一段简洁的摘要。",
    "请将以下文章内容总结为一段简洁的摘要。",
    "请将以下文章创建一个吸引人的标题。",
    "能否帮助将这篇文章重写为摘要形式？同时，需要一个能够准确反映主题的标题。",
    "需要将附加的文章内容精简成摘要，并提供一个简洁明了的标题。",
    "请帮忙把这篇详细的文章转换成一个简洁的摘要，并为之设计一个符合内容的引人注意的标题。",
    "我需要将这篇文章缩减为一个摘要，还需要一个既精确又引人注目的标题，你能帮忙吗？",
    "请提炼这篇文章的主要内容，制作一个摘要，并给出一个相关的、有吸引力的标题。",
    "能帮我把这篇长文章改写成简短的摘要形式，并配上一个适合的标题吗？",
    "请根据这篇文章内容，提炼出核心信息，撰写一个简洁的摘要并起一个贴切的标题。",
    "你能将这篇文章浓缩成一个摘要，并且创建一个表达文章主题的标题吗？",
    "请帮忙将文章内容转化为一个包含关键信息的摘要，并提供一个准确描述内容的标题。",
    "这篇文章讲了什么？",
    "帮我看看下面这篇文章讲的什么内容。",
    "下面的文章写的是什么？",
    "帮我阅读下面的文章，总结一些内容。",
    "如下的文章讲了什么内容？"
]

dy_text=["文章","段落","短文","新闻","片段"]
dy_action=["概括","总结","提取","归纳"]
dy_location=["下","上"]
dy_answer=["当然，","好，","好的，","收到，","明白，",""]

answers=[
    "当然，这段文章可以总结如下。",
    "当然，以下是文章总结的内容。",
    "当然，这段文章可以总结为一个标题，让我来帮您。",
    "当然，总结文章如下。",
    "当然，让我为你总结一下文章。",
    "当然，总结如下。",
    "当然，可以将文章总结为下面内容。",
    "当然，阅读文章后根据理解，讲述的内容总结为：",
    "当然，根据你的要求，我的总结如下。",
    "当然，我将上述文章的总结如下：",
    "当然，根据我的理解，总结文章如下：",
    "当然，总结：",
    "当然，上述文章可以总结为：",
    "当然，这篇文章可以总结为："
]


class CSLDataset(Dataset):
    def __init__(self,json_path,tokenizer_path,max_len):
        self.summary_list=[]
        with open(json_path,"r",encoding="utf-8") as file:
            for line in file:
                json_data=json.loads(line)
                content=json_data["content"].replace(".","。").replace(",","，")
                summary=json_data["summary"].replace(".","。").replace(",","，")+"。"

                chat_text="人类："

                question=questions[random.randint(0,len(questions)-1)]#随机选一个问
                answer=answers[random.randint(0,len(answers)-1)]#随机选一个答


    
                #上下文位置
                if random.randint(0,1)==0:#下，不做处理
                    chat_text+=question
                    chat_text+=content
                else:
                    chat_text+=content
                    chat_text+=question.replace("下","上")
                
                chat_text+="助手："

                #随机回答
                chat_text+=answer.replace("当然，",dy_answer[random.randint(0,len(dy_answer)-1)])
                #摘要
                chat_text+=summary
                #随机名词
                chat_text=chat_text.replace("文章",dy_text[random.randint(0,len(dy_text)-1)])
                #随机动词
                chat_text=chat_text.replace("总结",dy_action[random.randint(0,len(dy_action)-1)])

                self.summary_list.append(chat_text)

        self.tokenizer=Tokenizer(tokenizer_path)
        self.max_len=max_len

    def __getitem__(self, index):
        texts=self.summary_list[index]
        tokens=self.tokenizer.encode(texts,bos=True,eos=True)

        if(len(tokens)>self.max_len):
            tokens=tokens[:self.max_len-1]+[self.tokenizer.eos_id]

        token_tensor=torch.tensor(tokens, dtype=torch.long, device="cuda")
        return token_tensor[:-1].detach().clone(),token_tensor[1:].detach().clone()
    

if __name__=="__main__":
    csl_dataset=CSLDataset(r"D:\work\Datasets\CSL\train.json",r"weight\tokenizer.model",max_len=512)
    for a,b in csl_dataset:
        print(a.shape)