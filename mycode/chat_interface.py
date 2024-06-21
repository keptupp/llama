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


        pre_tokens=model(before,prev_pos=2)

        pre_tokens=torch.argmax(torch.softmax(pre_tokens,dim=-1),dim=-1)

        pre_tokens=pre_tokens.cpu().tolist()
        after=after.cpu().tolist()

        pre_text_list=[model.tokenizer.decode(pre_tokens[i]) for i in range(len(pre_tokens))]
        true_text_list=[model.tokenizer.decode(after[i]) for i in range(len(after))]
        



        print(pre_text_list)
        print(true_text_list)

        break

    

if __name__=="__main__":
    model = Llama.build(
        ckpt_dir="weight\pre_train\epoch_24.pt",
        tokenizer_path="weight/tokenizer_chinese.model",
        max_seq_len=512,
        max_batch_size=8,
    ).to(config.device)

    pre_dataset=PreTrainDataset(r"D:\work\Datasets\MNBVC\20230196\github.20230196.3.鏂伴椈\11.jsonl",r"weight/tokenizer.model",min_len=32,max_len=256)
    pre_dataloader=DataLoader(pre_dataset,batch_size=2,shuffle=True,collate_fn=my_collate_fn)

    dict_data=dict()
    dict_data["pretraindataloader"]=pre_dataloader


    chat_epoch(model,dict_data)