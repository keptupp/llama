import sys,os
sys.path.append(os.getcwd())
import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
import pandas as pd
from llama.tokenizer import Tokenizer
import json
import config

class PreTrainDataset(Dataset):
    def __init__(self,text_path,tokenizer_path,min_len,max_len):
        data=pd.read_json(text_path,lines=True)
        self.data_section=[]
        for row in data.itertuples():
            for one in getattr(row,'段落'):
                if(len(one["内容"])>min_len):
                    if (len(one["内容"])<max_len):
                        self.data_section.append(one["内容"])
                    else:
                        self.data_section.append(one["内容"][:max_len])

        self.tokenizer=Tokenizer(tokenizer_path)

    def __getitem__(self, index):
        token=self.tokenizer.encode(self.data_section[index],bos=True,eos=True)
        token=torch.tensor(token, dtype=torch.long, device="cuda")
        return token[:-1].detach().clone(),token[1:].detach().clone()

    def __len__(self):
        return len(self.data_section)
    
class WikiDataset(Dataset):
    def __init__(self,text_path,tokenizer_path,min_len,max_len):
        self.tokenizer=Tokenizer(tokenizer_path)

        self.file_list=self.get_filelist(text_path)
        self.read_index_flie=-1
        self.read_index_text=0
        self.read_next=False

        self.text_data=[]

        self.total_data=0
        self.min_len=min_len
        self.max_len=max_len
        

    def get_new_data(self):
        self.text_data.clear()
        self.read_index_flie+=1
        with open(self.file_list[self.read_index_flie], "r",encoding="utf-8") as fileHandler:
            while True:
                line = fileHandler.readline()
                if not line:
                    break
                text_list=json.loads(line.strip())["text"].split("\n")
                text_list=[one for one in text_list if len(one)>self.min_len]
                self.text_data+=text_list

        # self.total_data+=len(self.text_data)
        # print(len(self.text_data),self.total_data)

    def __getitem__(self, index):
        if(self.read_index_text>=len(self.text_data)):
            self.get_new_data()
            self.read_index_text=0

        text=self.text_data[self.read_index_text]
        self.read_index_text+=1
        token=self.tokenizer.encode(text,bos=True,eos=True)[:self.max_len]
        token=torch.tensor(token, dtype=torch.long, device="cuda")
        return token[:-1].detach().clone(),token[1:].detach().clone()

    def __len__(self):
        return 3300000
                

    def get_filelist(self,dir_path, file_list=[]):
        if os.path.isfile(dir_path):
            file_list.append(dir_path)
        elif os.path.isdir(dir_path):
            for item in os.listdir(dir_path):
                new_path = os.path.join(dir_path, item)
                self.get_filelist(new_path, file_list)
        return file_list

if __name__=="__main__":
    # pre_dataset=PreTrainDataset(r"/home/liuzheng/Data/MNBVC/20230196/github.20230196/11.jsonl",r"weight/tokenizer.model",min_len=32)
    # pre_dataloader=DataLoader(pre_dataset,batch_size=1,shuffle=True)
    # print(len(pre_dataset))
    # for token in pre_dataloader:
    #     print(token)

    wiki_dataset=WikiDataset(r"/home/liuzheng/Data/wiki_zh_2019/wiki_zh",r"weight/tokenizer.model",32,256)

    for wiki_dataset in wiki_dataset:
        pass