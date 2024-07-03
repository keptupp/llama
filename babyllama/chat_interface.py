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
import wikipedia
import zhconv

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



def chat_epoch(model,dict_data):
    total_token=[]
    reference=""
    text=input("资料：")
    wikipedia.set_lang("zh")
    # search_list=wikipedia.search(text)
    # print(search_list)
    # for index,one in enumerate(search_list[:3]):
    if(len(text)!=0):
        reference+=wikipedia.summary(text,sentences=10)
    reference=zhconv.convert(reference, 'zh-hans')
    print("资料：",len(reference),reference)
    reference_token=model.tokenizer.encode("资料:"+reference,bos=True,eos=True)
    
    first=True
    while True:
        print()
        text=input("人类: ")

        if first:#第一次对话，搜索资料
            total_token=total_token+model.tokenizer.encode("人类: "+text+"\n\n助手: ",bos=False,eos=False)
            first=False
        else:
            total_token=total_token+model.tokenizer.encode("\n\n人类: "+text+"\n\n助手: ",bos=False,eos=False)

        token=torch.tensor(reference_token+total_token, dtype=torch.long, device="cuda").unsqueeze(0)

        pre_tokens=model.inference(token,prev_pos=0,max_length=256,top_p=0.5)


        pre_text_list=[model.tokenizer.decode(pre_tokens[i]) for i in range(len(pre_tokens))]
        
        print("助手: ",pre_text_list[0])
        print()

        pre_token=model.tokenizer.encode(pre_text_list[0],bos=False,eos=False)
        total_token=total_token+pre_token

        
def shell_epoch(model,dict_data):
    
    while True:
        text=input("prompt:")
        token=model.tokenizer.encode(text,bos=True,eos=True)
        token=torch.tensor(token, dtype=torch.long, device="cuda").unsqueeze(0)

        pre_tokens=model.inference(token[:,:-1],prev_pos=0,max_length=256)


        pre_text_list=[model.tokenizer.decode(pre_tokens[i]) for i in range(len(pre_tokens))]
        
        print(pre_text_list)
        print()

if __name__=="__main__":
    model = Llama.build(
        tokenizer_path="weight/tokenizer.model",
        max_seq_len=2048,
        max_batch_size=8,
    ).to(config.device)

    model.load_state_dict(torch.load("weight/belle_epoch_3.pt"))

    dict_data=dict()


    # wikipedia.set_lang("zh")
    # print(wikipedia.search("减肥"))


    chat_epoch(model,dict_data)
    # shell_epoch(model,dict_data)