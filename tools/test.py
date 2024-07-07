import json
nums=0
with open(r"D:\work\Datasets\sft_data_zh.jsonl", "r" , encoding="utf-8") as file:
    for line in file:
        json.loads(line)
        nums+=1
        if nums%1e4==0:
            print(nums)

        if nums>(220978*16):
            print(line)

print(nums)