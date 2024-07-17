import sys,os
sys.path.append(os.getcwd())
import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
import pandas as pd
from llama.tokenizer import Tokenizer
import config
import torchvision.transforms as transforms

from tqdm import tqdm
import cv2
import json



class Vision_Chat_Dataset(Dataset):
    def __init__(self,text_path,image_path,tokenizer_path,min_len,max_len,):
       

        self.tokenizer=Tokenizer(tokenizer_path)
        self.max_len=max_len
        self.image_path=image_path
        self.norm=transforms.Compose([
            transforms.ToTensor(),  # 将图像转换为张量
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.len_total=0

        with open(text_path, "r" , encoding="utf-8") as file:
            self.data = json.load(file)        

    def __getitem__(self, index):
        src_data=self.data[index]

        image_path=self.image_path+"/"+src_data["image"]
        text_list=src_data["conversations"]
        total_token=[]
        
        # 处理文本
        for one in text_list:
            if one["from"]=="human":
                text="人类："+one["value"].replace('<image>\n',"").replace("\n<image>","")
                total_token+=self.tokenizer.encode(text,bos=False,eos=False)
            else:
                total_token+=self.tokenizer.encode("助手："+one["value"],bos=False,eos=True)

        total_token=[self.tokenizer.bos_id]+total_token#加上开始符号
        
        image=cv2.imread(image_path)
        h,w,c=image.shape
        ratio=512/max([h,w])
        new_h,new_w=int(round(h*ratio)),int(round(w*ratio))
        image=cv2.resize(image,[new_w,new_h])

        image = cv2.copyMakeBorder(image, 0, 512-new_h, 0, 512-new_w, cv2.BORDER_CONSTANT, None, (0, 0, 0))
        
        image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        image=self.norm(image)

        # cv2.imshow("test",image.permute(1,2,0).numpy())
        # cv2.waitKey(2000)
        

        if(len(total_token)>self.max_len):
            total_token=total_token[:self.max_len-1]#截断后就不加结束符了
        token=torch.tensor(total_token, dtype=torch.long, device="cuda")
        
        return token[:-1].detach().clone(),token[1:].detach().clone(),image.to(config.device)

    def __len__(self):
        return len(self.data)
    

if __name__=="__main__":
    # pretrain_dataset=Pretrain_Dataset(r"D:\work\Datasets\Chinese-LLaVA-Vision-Instructions\LLaVA-CC3M-Pretrain-595K\chat-translated.json",r"D:\work\Datasets\Chinese-LLaVA-Vision-Instructions\LLaVA-CC3M-Pretrain-595K\images",r"weight\tokenizer.model",50,256)
    pretrain_dataset=Vision_Chat_Dataset(r"D:\work\Datasets\Chinese-LLaVA-Vision-Instructions\LLaVA-Instruct-150K\translated\llava_instruct_80k.json",r"D:\work\Datasets\sharegptv4\coco\train2017",r"weight\tokenizer.model",50,256)

    for _ in tqdm(pretrain_dataset):
        pass
