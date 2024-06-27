import sys,os
sys.path.append(os.getcwd())
from llama_class import Llama, Dialog
from pretrain_textdataset import PreTrainDataset
from chat_dataset import ChatDataset
from torch.utils.data import Dataset,DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import config
import numpy as np
from torch.utils.tensorboard import SummaryWriter

def my_collate_fn(batch):
    max_lenght=max([one[0].shape[0] for one in batch])
    # max_lenght=2048

    padding_before_token=None
    padding_after_token=None
    for one in batch:
        if padding_before_token is None:
            padding_before_token=torch.nn.functional.pad(one[0], (0,max_lenght-one[0].shape[0]), mode='constant', value=0).unsqueeze(0)
            padding_after_token=torch.nn.functional.pad(one[1], (0,max_lenght-one[0].shape[0]), mode='constant', value=0).unsqueeze(0)
        else:
            padding_before_token=torch.cat([padding_before_token,torch.nn.functional.pad(one[0], (0,max_lenght-one[0].shape[0]), mode='constant', value=0).unsqueeze(0)],dim=0)
            padding_after_token=torch.cat([padding_after_token,torch.nn.functional.pad(one[1], (0,max_lenght-one[0].shape[0]), mode='constant', value=0).unsqueeze(0)],dim=0)

    return padding_before_token,padding_after_token


nums=0
def train_epoch(model,dict_data):
    global nums
    bar=tqdm(dict_data["pretraindataloader"],ncols=100)

    total_loss=0

    for before,after in bar:
        dict_data["optimizer"].zero_grad()

        start_pos=np.random.randint(32)
        pre_tokens=model(before,start_pos)

        pre_tokens=pre_tokens.permute(0,2,1)


        loss=dict_data["crossentropyloss"](pre_tokens,after)

        mask=(after!=0).float()
        loss=(loss*mask).sum()/mask.sum()

        loss.backward()

        dict_data["optimizer"].step()
        dict_data["scheduler"].step()

        total_loss+=loss.item()
        bar.set_postfix(loss=loss.item())

        nums+=1
        if nums%16==0:
            dict_data["writer"].add_scalar('Loss/train', nums,loss.item())

    print("平均损失",total_loss/len(dict_data["pretraindataloader"]))

def train(model,dict_data):

    for epoch in range(1,dict_data["epoch"]+1):
        print()
        print("epoch",epoch)
        train_epoch(model,dict_data)

        torch.save(model.state_dict(),"weight/pre_train/epoch_"+str(epoch)+".pt")
        

if __name__=="__main__":
    model = Llama.build(
        tokenizer_path="weight/tokenizer.model",
        max_seq_len=2048,
        max_batch_size=8,
    ).to(config.device)
    model.load_state_dict(torch.load("weight/pre_train/chat_epoch_10.pt"))

    # pre_dataset=PreTrainDataset(r"/home/liuzheng/Data/MNBVC/20230196/github.20230196/11.jsonl",r"weight/tokenizer.model",min_len=32,max_len=256)
    # pre_dataloader=DataLoader(pre_dataset,batch_size=8,shuffle=True,collate_fn=my_collate_fn)

    data_text=dict()
    data_text["redgpt"]=r"/home/liuzheng/Data/RedGPT-Dataset-V1-CN.json"
    data_text["naturalconv"]=r"/home/liuzheng/Data/NaturalConv.json"
    data_text["pclue"]=[
        r"/home/liuzheng/Data/pCLUE/pCLUE_train_1.json",
        # r"/home/liuzheng/Data/pCLUE/pCLUE_train_2.json",
    ]
    chat_dataset=ChatDataset(data_text,r"weight/tokenizer.model",min_len=32,max_len=2048)
    chat_dataloader=DataLoader(chat_dataset,batch_size=4,shuffle=True,collate_fn=my_collate_fn)

    dict_data=dict()
    dict_data["pretraindataloader"]=chat_dataloader
    dict_data["crossentropyloss"]=nn.CrossEntropyLoss(reduction='none')

    dict_data["epoch"]=10
    dict_data["optimizer"] = optim.AdamW(model.parameters(), lr=1e-3)
    dict_data["scheduler"] = optim.lr_scheduler.CosineAnnealingLR(dict_data["optimizer"], T_max = dict_data["epoch"]*len(chat_dataloader),eta_min=1e-4)
    dict_data["writer"] = SummaryWriter('weight/log_tensorboard')


    train(model,dict_data)