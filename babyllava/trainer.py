import sys,os
sys.path.append(os.getcwd())
from torch.utils.data import Dataset,DataLoader,ConcatDataset
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import config
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from tools.metric import Metric
from babyllava import BabyLLaVa
from datasets.vision_chat_dataset import Vision_Chat_Dataset


def my_collate_fn(batch):
    max_lenght=max([one[0].shape[0] for one in batch])

    padding_before_token=None
    padding_after_token=None
    images=None
    for one in batch:
        if padding_before_token is None:
            padding_before_token=torch.nn.functional.pad(one[0], (0,max_lenght-one[0].shape[0]), mode='constant', value=0).unsqueeze(0)
            padding_after_token=torch.nn.functional.pad(one[1], (0,max_lenght-one[0].shape[0]), mode='constant', value=0).unsqueeze(0)
            images=one[2].unsqueeze(0)
        else:
            padding_before_token=torch.cat([padding_before_token,torch.nn.functional.pad(one[0], (0,max_lenght-one[0].shape[0]), mode='constant', value=0).unsqueeze(0)],dim=0)
            padding_after_token=torch.cat([padding_after_token,torch.nn.functional.pad(one[1], (0,max_lenght-one[0].shape[0]), mode='constant', value=0).unsqueeze(0)],dim=0)
            images=torch.cat([images,one[2].unsqueeze(0)],dim=0)

    return padding_before_token,padding_after_token,images

@torch.no_grad()
def val_epoch(model,dict_data):
    total_loss=0

    bar=tqdm(dict_data["valdataloader"],ncols=100)
    for before,after in bar:
        pre_tokens=model(before,0)
        pre_tokens=pre_tokens.permute(0,2,1)
        loss=dict_data["crossentropyloss"](pre_tokens,after)
        mask=(after!=0).float()
        loss=(loss*mask).sum()/mask.sum()

        dict_data["metric"].update_acc(pre_tokens,after)

        bar.set_postfix(loss=loss.item(),acc=dict_data["metric"].get_acc())
        total_loss+=loss.item()
        
    print("平均损失",total_loss/len(dict_data["valdataloader"]),"精确度acc",dict_data["metric"].get_acc())


nums=0
def train_epoch(model,dict_data):
    global nums
    bar=tqdm(dict_data["pretraindataloader"],ncols=100)

    total_loss=0

    for before,after,image in bar:
        dict_data["optimizer"].zero_grad()

        start_pos=38#前38和token是图片，不需要学习
        pre_tokens=model(before,image,start_pos)

        pre_tokens=pre_tokens.permute(0,2,1)[:,:,start_pos:]#去掉前38个图片的预测token

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
            torch.save(model.state_dict(),"weight/multimodal/temp.pt")

    print("平均损失",total_loss/len(dict_data["pretraindataloader"]))

def train(model,dict_data):

    for epoch in range(1,dict_data["epoch"]+1):
        print()
        print("epoch",epoch)
        train_epoch(model,dict_data)

        # val_epoch(model,dict_data)

        torch.save(model.state_dict(),"weight/multimodal/epoch_"+str(epoch)+".pt")
        

if __name__=="__main__":
    model = BabyLLaVa.build(tokenizer_path="weight/tokenizer.model").to(config.device)
    # model.load_state_dict(torch.load("weight/pre_train/pretrain_deepctrl_sft_epoch_1.pt"))


    text_path=r"D:\work\Datasets\Chinese-LLaVA-Vision-Instructions\LLaVA-Instruct-150K\translated\llava_instruct_80k.json"
    image_path=r"D:\work\Datasets\sharegptv4\coco\train2017"
    tokenizer_path=r"weight\tokenizer.model"
    vision_chat_dataset=Vision_Chat_Dataset(text_path,image_path,tokenizer_path,0,512)




    train_dataset=ConcatDataset([vision_chat_dataset])
    train_dataloader=DataLoader(train_dataset,batch_size=2,shuffle=True,collate_fn=my_collate_fn)


    dict_data=dict()
    dict_data["metric"]=Metric()
    dict_data["pretraindataloader"]=train_dataloader
    dict_data["valdataloader"]=train_dataloader
    dict_data["crossentropyloss"]=nn.CrossEntropyLoss(reduction='none')

    dict_data["epoch"]=3
    dict_data["optimizer"] = optim.AdamW(model.parameters(), lr=1e-3)
    dict_data["scheduler"] = optim.lr_scheduler.CosineAnnealingLR(dict_data["optimizer"], T_max = dict_data["epoch"]*len(train_dataloader),eta_min=1e-4)
    dict_data["writer"] = SummaryWriter('weight/log_tensorboard/step9_multimodal')


    train(model,dict_data)