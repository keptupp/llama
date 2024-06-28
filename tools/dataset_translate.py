from deep_translator import GoogleTranslator
from translate import Translator
import pandas as pd
import json
from tqdm import tqdm
from argostranslate import package, settings, translate
proxies={
    "http":"127.0.0.1:7890",
    "https":"127.0.0.1:7890"
}
# translator = Translator(to_lang="zh")
# translated_text = GoogleTranslator(source='en', target='zh-CN',proxies=proxies).translate(text)
from_code = 'en'
to_code = 'zh'
filename = f'translate-{from_code}_{to_code}.argosmodel'
path = settings.downloads_dir / filename
package.install_from_path(path)
settings.device="cuda"


def traslate(text):
    translated_text=""
    if(len(text)>=499):
        text=text.split(". ")
        for one in text:
            translated_text+=translate.translate(one, from_code, to_code)
    else:
        translated_text+=translate.translate(text, from_code, to_code)
    return translated_text

def read_data(data_path):
    data=pd.read_json(data_path,lines=True)
    nums=0
    with open(r"D:\work\Datasets\grade-school-math\test_chinese.jsonl", 'w',encoding='utf-8') as f:
        for row in data.itertuples():
            question=getattr(row,"question")
            answer=getattr(row,"answer")
            nums+=1
            print(nums)

            question=traslate(question)
            answer=traslate(answer)

            item=dict()
            item["question"]=question
            item["answer"]=answer
            f.write(json.dumps(item,ensure_ascii=False) + '\n')

if __name__=="__main__":

    data_path=r"D:\work\Datasets\grade-school-math\test.jsonl"
    read_data(data_path)
