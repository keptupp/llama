import sys,os
sys.path.append(os.getcwd())
import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
import pandas as pd
from llama.tokenizer import Tokenizer
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
        print(12)
    

if __name__=="__main__":
    pre_dataset=PreTrainDataset(r"/home/liuzheng/Data/MNBVC/20230196/github.20230196/11.jsonl",r"weight/tokenizer.model",min_len=32)
    pre_dataloader=DataLoader(pre_dataset,batch_size=1,shuffle=True)
    print(len(pre_dataset))
    for token in pre_dataloader:
        print(token)