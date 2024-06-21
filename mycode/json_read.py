import sys,os
sys.path.append(os.getcwd())
import pandas as pd
from llama.tokenizer import Tokenizer



def read_json_file(json_path,):
    data=pd.read_json(json_path,lines=True)
    data_section=[]
    for row in data.itertuples():
        # print(row)
        for one in getattr(row,'段落'):
            if(len(one["内容"])>32):
                data_section.append(one["内容"])
    return data_section

if __name__=="__main__":
    data=read_json_file(r"D:\work\Datasets\MNBVC\20230196\github.20230196.3.鏂伴椈\11.jsonl")
    tokenizer=Tokenizer("weight/tokenizer_chinese.model")

    for text in data:
        print("原始文本",text)
        token=tokenizer.encode(text, bos=True, eos=True)
        print("转token",token)
        print(token)
        de_text=tokenizer.decode(token)
        print("转text",de_text)

        print("===========================================================")