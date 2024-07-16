import json
import cv2
from tqdm import tqdm
import os
nums=0
with open(r"D:\work\Datasets\sharegptv4\sharegpt4v_instruct_gpt4-vision_cap100k.json", "r" , encoding="utf-8") as file:
    data = json.load(file)

    map_data=dict()
    for one in tqdm(data):
        dataset_name=one["image"].split("/")[0]
        if dataset_name not in map_data:
            map_data[dataset_name]=1
        else:
            map_data[dataset_name]+=1
        
        if not os.path.exists("D:/work/Datasets/sharegptv4/"+one["image"]):
            if dataset_name=="sam" or dataset_name=="llava" or dataset_name=="wikiart":
                pass
            else:
                print("D:/work/Datasets/sharegptv4/"+one["image"])
                break
            

    
    for key in map_data.keys():
        print(key,map_data[key])