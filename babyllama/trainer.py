import sys,os
sys.path.append(os.getcwd())
from llama_class import Llama, Dialog
from pretrain_textdataset import PreTrainDataset,WikiDataset
from chat_dataset import ChatDataset,BigChatDataset,DeepctrlDataset
from torch.utils.data import Dataset,DataLoader,ConcatDataset
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import config
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from tools.metric import Metric
from csl_dataset import CSLDataset

# def my_collate_fn(batch):
#     max_lenght=max([one[0].shape[0] for one in batch])
#     padding_before_token=None
#     padding_after_token=None
#     for one in batch:
#         if padding_before_token is None:
#             padding_before_token=torch.nn.functional.pad(one[0], (0,max_lenght-one[0].shape[0]), mode='constant', value=0).unsqueeze(0)
#             padding_after_token=torch.nn.functional.pad(one[1], (0,max_lenght-one[0].shape[0]), mode='constant', value=0).unsqueeze(0)
#         else:
#             padding_before_token=torch.cat([padding_before_token,torch.nn.functional.pad(one[0], (0,max_lenght-one[0].shape[0]), mode='constant', value=0).unsqueeze(0)],dim=0)
#             padding_after_token=torch.cat([padding_after_token,torch.nn.functional.pad(one[1], (0,max_lenght-one[0].shape[0]), mode='constant', value=0).unsqueeze(0)],dim=0)
#     return padding_before_token,padding_after_token
def my_collate_fn(batch):
    max_lenght=max([one[0].shape[0] for one in batch])

    padding_before_token=None
    padding_after_token=None
    global_w=None
    i=None
    for one in batch:
        if padding_before_token is None:
            padding_before_token=torch.nn.functional.pad(one[0], (0,max_lenght-one[0].shape[0]), mode='constant', value=0).unsqueeze(0)
            padding_after_token=torch.nn.functional.pad(one[1], (0,max_lenght-one[0].shape[0]), mode='constant', value=0).unsqueeze(0)
            global_w=torch.nn.functional.pad(one[2], (0,4-one[2].shape[0]), mode='constant', value=0).unsqueeze(0)
            i=one[3].unsqueeze(0)
        else:
            padding_before_token=torch.cat([padding_before_token,torch.nn.functional.pad(one[0], (0,max_lenght-one[0].shape[0]), mode='constant', value=0).unsqueeze(0)],dim=0)
            padding_after_token=torch.cat([padding_after_token,torch.nn.functional.pad(one[1], (0,max_lenght-one[0].shape[0]), mode='constant', value=0).unsqueeze(0)],dim=0)
            global_w=torch.cat([global_w,torch.nn.functional.pad(one[2], (0,4-one[2].shape[0]), mode='constant', value=0).unsqueeze(0)],dim=0)
            i=torch.cat([i,one[3].unsqueeze(0)],dim=0)
    return padding_before_token,padding_after_token,global_w,i

nums_val=0
@torch.no_grad()
def val_epoch(model,dict_data):
    global nums_val
    total_loss=0
    metric=Metric()

    bar=tqdm(dict_data["valdataloader"],ncols=100)
    for before,after,g_w,m_l in bar:
        nums_val+=1
        pre_tokens=model(before,0)
        pre_tokens=pre_tokens.permute(0,2,1)
        loss=dict_data["crossentropyloss"](pre_tokens,after)
        mask=(after!=0).float()
        loss=(loss*mask).sum()/mask.sum()

        dict_data["metric"].update_acc(pre_tokens,after)

        bar.set_postfix(loss=loss.item(),acc=dict_data["metric"].get_acc())
        total_loss+=loss.item()

        if nums_val%50==0:
            dict_data["writer"].add_scalar('Val_Loss',loss.item(),nums_val)
            metric.update_acc(pre_tokens,after)
            acc=metric.get_acc()

            ppl=metric.get_ppl(pre_tokens,after)
            dict_data["writer"].add_scalar('Val_PPL',ppl,nums_val)
            dict_data["writer"].add_scalar('Val_Acc',acc,nums_val)
        
    print("平均损失",total_loss/len(dict_data["valdataloader"]),"精确度acc",dict_data["metric"].get_acc())


nums=0
def train_epoch(model,dict_data):
    global nums
    bar=tqdm(dict_data["pretraindataloader"],ncols=100)

    total_loss=0

    for before,after,g_w,m_l in bar:
        dict_data["optimizer"].zero_grad()

        # start_pos=np.random.randint(32)
        start_pos=0
        pre_tokens=model(before,start_pos,m_l)

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
        if nums%1e2==0:
            dict_data["writer"].add_scalar('Train_Loss',loss.item(),nums)
            dict_data["metric"].update_acc(pre_tokens,after)
            acc=dict_data["metric"].get_acc()

            ppl=dict_data["metric"].get_ppl(pre_tokens,after)
            dict_data["writer"].add_scalar('Train_PPL',ppl,nums)
            dict_data["writer"].add_scalar('Train_Acc',acc,nums)
        if nums%1e4==0:
            torch.save(model.state_dict(),"weight/pre_train/temp.pt")

    print("平均损失",total_loss/len(dict_data["pretraindataloader"]))

def train(model,dict_data):

    for epoch in range(1,dict_data["epoch"]+1):
        print()
        print("epoch",epoch)
        train_epoch(model,dict_data)

        val_epoch(model,dict_data)

        torch.save(model.state_dict(),"weight/pre_train/epoch_"+str(epoch)+".pt")
        

if __name__=="__main__":
    model = Llama.build(
        tokenizer_path="weight/tokenizer.model",
        max_seq_len=2048,
        max_batch_size=8,
    ).to(config.device)
    # model.load_state_dict(torch.load("weight/pre_train/pretrain_deepctrl_sft_epoch_1.pt"),strict=False)
    # model.load_state_dict(torch.load("weight/pre_train/epoch_10.pt"),strict=True)

    # pre_dataset=PreTrainDataset(r"/home/liuzheng/Data/MNBVC/20230196/github.20230196/11.jsonl",r"weight/tokenizer.model",min_len=32,max_len=256)
    # pre_dataloader=DataLoader(pre_dataset,batch_size=8,shuffle=True,collate_fn=my_collate_fn)

    data_text=dict()
    # data_text["redgpt"]=r"/home/liuzheng/Data/RedGPT-Dataset-V1-CN.json"
    # data_text["naturalconv"]=r"/home/liuzheng/Data/NaturalConv.json"
    # data_text["redgpt0_8M"]=r"/home/liuzheng/Data/multiturn_chat_0.8M.json"
    # chat_dataset=ChatDataset(data_text,r"weight/tokenizer.model",min_len=32,max_len=512)
    # chat_dataloader=DataLoader(chat_dataset,batch_size=16,shuffle=True,collate_fn=my_collate_fn)

    # wiki_dataset=WikiDataset(r"/home/liuzheng/Data+/wiki_zh_2019/wiki_zh",r"weight/tokenizer.model",32,256)
    # wiki_dataloader=DataLoader(wiki_dataset,batch_size=32,shuffle=True,collate_fn=my_collate_fn)

    # belle_dataset=BigChatDataset(r"/home/liuzheng/Data/Belle_open_source_0.5M.json",r"weight/tokenizer.model",256)
    # belle_dataloader=DataLoader(belle_dataset,batch_size=32,shuffle=False,collate_fn=my_collate_fn)

    # deepctrldataset=DeepctrlDataset(r"/home/liuzheng/Data/sft_data_zh.jsonl",r"weight/tokenizer.model",512)
    # deepctrl_dataloader=DataLoader(deepctrldataset,batch_size=16,shuffle=False,collate_fn=my_collate_fn)

    csl_dataset=CSLDataset(r"D:/work/Datasets/CSL/train.json",r"weight/tokenizer.model",max_len=512)
    # csl_dataloader=DataLoader(chat_dataset,batch_size=16,shuffle=True,collate_fn=my_collate_fn)

    csl_val_dataset=CSLDataset(r"D:/work/Datasets/CSL/val.json",r"weight/tokenizer.model",max_len=512)


    train_dataset=ConcatDataset([csl_dataset])
    train_dataloader=DataLoader(train_dataset,batch_size=2,shuffle=True,collate_fn=my_collate_fn)
    
    csl_val_dataloader=DataLoader(csl_val_dataset,batch_size=2,shuffle=False,collate_fn=my_collate_fn)



    dict_data=dict()
    dict_data["metric"]=Metric()
    dict_data["pretraindataloader"]=train_dataloader
    dict_data["valdataloader"]=csl_val_dataloader
    dict_data["crossentropyloss"]=nn.CrossEntropyLoss(reduction='none')

    dict_data["epoch"]=10
    dict_data["optimizer"] = optim.AdamW(model.parameters(), lr=1e-3)
    dict_data["scheduler"] = optim.lr_scheduler.CosineAnnealingLR(dict_data["optimizer"], T_max = dict_data["epoch"]*len(train_dataloader),eta_min=1e-4)
    dict_data["writer"] = SummaryWriter('weight/log_tensorboard/step20_gl_csl_new4')


    train(model,dict_data)