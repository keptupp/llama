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
import torchvision.transforms as transforms
import cv2

norm=transforms.Compose([
    transforms.ToTensor(),  # 将图像转换为张量
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def chat_epoch(model,dict_data):
    total_token=[model.tokenizer.bos_id]

    image_path=dict_data["image_path"]
    image=cv2.imread(image_path)
    h,w,c=image.shape
    ratio=512/max([h,w])
    new_h,new_w=int(round(h*ratio)),int(round(w*ratio))
    image=cv2.resize(image,[new_w,new_h])

    image = cv2.copyMakeBorder(image, 0, 512-new_h, 0, 512-new_w, cv2.BORDER_CONSTANT, None, (0, 0, 0))
    
    image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    image=norm(image).unsqueeze(0).to(config.device)

    while True:
        print()
        text=input("人类: ")
        total_token=total_token+model.tokenizer.encode("人类："+text+"助手：",bos=False,eos=False)
        # total_token=total_token+model.tokenizer.encode("人类："+text,bos=False,eos=False)

        token=torch.tensor(total_token, dtype=torch.long, device="cuda").unsqueeze(0)

        pre_tokens=model.inference(token,image,prev_pos=38,max_length=512,top_p=0)

        pre_text_list=[model.tokenizer.decode(pre_tokens[i]) for i in range(len(pre_tokens))]
        
        print("助手：")
        print(pre_text_list[0])
        print()

        pre_token=model.tokenizer.encode(pre_text_list[0],bos=False,eos=True)
        total_token=total_token+pre_token

if __name__=="__main__":
    model = BabyLLaVa.build(
        tokenizer_path="weight/tokenizer.model",
    ).to(config.device)

    # model.load_state_dict(torch.load("weight\multi_chat_epoch_1.pt"))
    model.load_state_dict(torch.load("weight/multimodal/epoch_3.pt"))

    dict_data=dict()
    dict_data["image_path"]=r"/home/liuzheng/Data/Chinese-LLaVA-Vision-Instructions/LLaVA-CC3M-Pretrain-595K/images/GCC_train_000002794.jpg"
    # dict_data["image_path"]=r"/home/liuzheng/Data/sharegpt4v/train2017/000000001448.jpg"

    # wikipedia.set_lang("zh")
    # print(wikipedia.search("减肥"))


    chat_epoch(model,dict_data)
    # shell_epoch(model,dict_data)