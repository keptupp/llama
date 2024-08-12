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
from csl_dataset import CSLDataset_Eval
from tools.metric_rouge import RougeUtil

def my_collate_fn(batch):
    # print(batch)
    max_lenght=max([one[0].shape[0] for one in batch])
    # max_lenght=256

    padding_before_token=None
    padding_after_token=None
    for one in batch:
        if padding_before_token is None:
            padding_before_token=torch.nn.functional.pad(one[0], (0,max_lenght-one[0].shape[0]), mode='constant', value=0).unsqueeze(0)
            if isinstance(one[1], str):
                padding_after_token=one[1]
            else:
                padding_after_token=torch.nn.functional.pad(one[1], (0,max_lenght-one[0].shape[0]), mode='constant', value=0).unsqueeze(0)
        else:
            padding_before_token=torch.cat([padding_before_token,torch.nn.functional.pad(one[0], (0,max_lenght-one[0].shape[0]), mode='constant', value=0).unsqueeze(0)],dim=0)
            if isinstance(one[1], str):
                padding_after_token=one[1]
            else:
                padding_after_token=torch.cat([padding_after_token,torch.nn.functional.pad(one[1], (0,max_lenght-one[0].shape[0]), mode='constant', value=0).unsqueeze(0)],dim=0)

    return padding_before_token,padding_after_token

def eval_one(model,dict_data):

    metric_rouge=RougeUtil()
    nums=0

    bar=tqdm(dict_data["eval_dataloader"],ncols=100)

    for token,answer in bar:
        pre_tokens=model.inference(token,prev_pos=0,max_length=512,top_p=0)

        pre_text_list=[model.tokenizer.decode(pre_tokens[i]) for i in range(len(pre_tokens))]


        if len(pre_text_list[0].split("："))==2:
            pre=pre_text_list[0].split("：")[1]
        elif len(pre_text_list[0].split("。"))>=2:
            pre=pre_text_list[0].split("。")[1]
        else:
            pre=pre_text_list[0]
            # print(pre)
        
        if len(pre)==0:
            pre=pre_text_list[0]
            # print(pre)
        if len(pre)==0:
            pre="你"
            # print("输出空")



        
        ref=answer[0][:len(answer[0])-1]

        metric_rouge.update_rouge(pre,ref)
        nums+=1

        if nums%10==0:
            json_rouge=metric_rouge.get_rouge()
            bar.set_postfix(rouge_1=json_rouge["rouge-1"]["f"],rouge_2=json_rouge["rouge-2"]["f"],rouge_L=json_rouge["rouge-l"]["f"])
    print(metric_rouge.get_rouge())

def eval_batch(model,dict_data):

    metric_rouge=RougeUtil()
    nums=0

    bar=tqdm(dict_data["eval_dataloader"],ncols=100)

    for token,answer in bar:
        pre_tokens=model.batch_inference(token,prev_pos=0,max_length=512,top_p=0)

        pre_text_list=[model.tokenizer.decode(pre_tokens[i]) for i in range(len(pre_tokens))]


        if len(pre_text_list[0].split("："))==2:
            pre=pre_text_list[0].split("：")[1]
        elif len(pre_text_list[0].split("。"))>=2:
            pre=pre_text_list[0].split("。")[1]
        else:
            pre=pre_text_list[0]
            # print(pre)
        
        if len(pre)==0:
            pre=pre_text_list[0]
            # print(pre)
        if len(pre)==0:
            pre="你"
            # print("输出空")
        
        ref=answer[0][:len(answer[0])-1]

        metric_rouge.update_rouge(pre,ref)
        nums+=1

        if nums%10==0:
            json_rouge=metric_rouge.get_rouge()
            bar.set_postfix(rouge_1=json_rouge["rouge-1"]["f"],rouge_2=json_rouge["rouge-2"]["f"],rouge_L=json_rouge["rouge-l"]["f"])
    print(metric_rouge.get_rouge())
        



def chat_epoch(model,dict_data):


    while True:
        print()
        text=input("人类: ")


        total_token=total_token+model.tokenizer.encode("人类："+text+"助手：",bos=False,eos=False)
        # total_token=total_token+model.tokenizer.encode("人类："+text,bos=False,eos=False)

        token=torch.tensor(total_token, dtype=torch.long, device="cuda").unsqueeze(0)

        pre_tokens=model.inference(token,prev_pos=0,max_length=256,top_p=0)

        pre_text_list=[model.tokenizer.decode(pre_tokens[i]) for i in range(len(pre_tokens))]
        
        print("助手：")
        print(pre_text_list[0])
        print()

        pre_token=model.tokenizer.encode(pre_text_list[0],bos=False,eos=True)
        total_token=total_token+pre_token
        # print(total_token)
        
if __name__=="__main__":
    model = Llama.build(
        tokenizer_path="weight/tokenizer.model",
        max_seq_len=2048,
        max_batch_size=8,
    ).to(config.device)

    model.load_state_dict(torch.load("weight/pre_train/maybe_csl_epoch_10.pt"))


    dict_data=dict()

    csl_dataset=CSLDataset_Eval(r"/home/liuzheng/Data/CSL/test.json",r"weight/tokenizer.model",max_len=512)
    train_dataloader=DataLoader(csl_dataset,batch_size=1,shuffle=False,collate_fn=None)

    dict_data["eval_dataloader"]=train_dataloader



    eval_one(model,dict_data)