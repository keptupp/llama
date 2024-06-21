import sys,os
sys.path.append(os.getcwd())
from llama_class import Llama, Dialog
from pretrain_textdataset import PreTrainDataset
from torch.utils.data import Dataset,DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import config
from torch.utils.tensorboard import SummaryWriter

def my_collate_fn(batch):
    # print(batch)
    max_lenght=max([one[0].shape[0] for one in batch])
    # max_lenght=256

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



def train_epoch(model,dict_data):
    bar=tqdm(dict_data["pretraindataloader"],ncols=100)
    nums=0
    for before,after in bar:
        dict_data["optimizer"].zero_grad()

        pre_tokens=model(before)

        pre_tokens=pre_tokens.permute(0,2,1)


        loss=dict_data["crossentropyloss"](pre_tokens,after)

        mask=(after!=0).float()
        loss=(loss*mask).sum()/mask.sum()

        loss.backward()

        dict_data["optimizer"].step()
        dict_data["scheduler"].step()

        bar.set_postfix(loss=loss.item())

        nums+=1
        if nums%16==0:
            dict_data["writer"].add_scalar('Loss/train', loss.item())



def train(model,dict_data):

    for epoch in range(1,dict_data["epoch"]+1):
        train_epoch(model,dict_data)

        torch.save(model.state_dict(),"weight/pre_train/epoch_"+str(epoch)+".pt")
        

if __name__=="__main__":
    model = Llama.build(
        ckpt_dir="",
        tokenizer_path="weight/tokenizer.model",
        max_seq_len=512,
        max_batch_size=8,
    ).to(config.device)

    pre_dataset=PreTrainDataset(r"/home/liuzheng/Data/MNBVC/20230196/github.20230196/11.jsonl",r"weight/tokenizer.model",min_len=32,max_len=256)
    pre_dataloader=DataLoader(pre_dataset,batch_size=8,shuffle=True,collate_fn=my_collate_fn)

    dict_data=dict()
    dict_data["pretraindataloader"]=pre_dataloader
    dict_data["crossentropyloss"]=nn.CrossEntropyLoss(reduction='none')

    dict_data["epoch"]=50
    dict_data["optimizer"] = optim.AdamW(model.parameters(), lr=1e-3)
    dict_data["scheduler"] = optim.lr_scheduler.CosineAnnealingLR(dict_data["optimizer"], T_max = dict_data["epoch"]*len(pre_dataloader),eta_min=3e-4)
    dict_data["writer"] = SummaryWriter('weight/log_tensorboard')


    train(model,dict_data)