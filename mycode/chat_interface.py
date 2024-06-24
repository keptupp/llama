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



def chat_epoch(model,dict_data):
    bar=tqdm(dict_data["pretraindataloader"],ncols=100)
    nums=0
    for before,after in bar:

        print(before.shape)
        pre_tokens=model.inference(before,prev_pos=0,max_length=256)

        # pre_tokens=torch.argmax(torch.softmax(pre_tokens,dim=-1),dim=-1)

        # pre_tokens=pre_tokens.cpu().tolist()
        after=after.cpu().tolist()

        pre_text_list=[model.tokenizer.decode(pre_tokens[i]) for i in range(len(pre_tokens))]
        true_text_list=[model.tokenizer.decode(after[i]) for i in range(len(after))]
        


        print()
        print(true_text_list)
        print(pre_text_list)
        print()

        
def shell_epoch(model,dict_data):
    while True:
        text=input("prompt：")
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
    model.load_state_dict(torch.load("weight/pre_train/pretrain.pt"))

    pre_dataset=PreTrainDataset(r"/home/liuzheng/Data/MNBVC/20230196/github.20230196/11.jsonl",r"weight/tokenizer.model",min_len=32,max_len=40)
    pre_dataloader=DataLoader(pre_dataset,batch_size=1,shuffle=True,collate_fn=my_collate_fn)

    dict_data=dict()
    dict_data["pretraindataloader"]=pre_dataloader


    # chat_epoch(model,dict_data)
    shell_epoch(model,dict_data)