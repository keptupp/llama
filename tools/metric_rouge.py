# from rouge import Rouge
from rouge_chinese import Rouge



class RougeUtil():
    def __init__(self):
        #记录总的结果
        self.rouge_dict={
            "rouge-1":{'r': 0, 'p': 0, 'f': 0},
            "rouge-2":{'r': 0, 'p': 0, 'f': 0},
            "rouge-l":{'r': 0, 'p': 0, 'f': 0}
        }
        #记录数量
        self.nums=0
        self.rouge=Rouge()
    
    #输入格式还有待商定
    def update_rouge(self,pre,ref):
        pre=" ".join([pre[i:i+1] for i in range(len(pre))])
        ref=" ".join([ref[i:i+1] for i in range(len(ref))])
        rouge_json=self.rouge.get_scores([pre], [ref],avg=True)
        self.nums+=1
        for key_r in self.rouge_dict.keys():
            for key in self.rouge_dict[key_r].keys():
                self.rouge_dict[key_r][key]+=rouge_json[key_r][key]

    
    def get_rouge(self):
        rouge_dict={
            "rouge-1":{'r': 0, 'p': 0, 'f': 0},
            "rouge-2":{'r': 0, 'p': 0, 'f': 0},
            "rouge-l":{'r': 0, 'p': 0, 'f': 0}
        }
        for key_r in self.rouge_dict.keys():
            for key in self.rouge_dict[key_r].keys():
                rouge_dict[key_r][key]=self.rouge_dict[key_r][key]/self.nums
        return rouge_dict

if __name__=="__main__":
    rouge_util=RougeUtil()
    pre="深海潜水器电源系统的研究现状分析"
    ref="深海潜水器电源系统的研究现"
    print(" ".join([pre[i:i+1] for i in range(len(pre))]))
    rouge_util.update_rouge(pre,ref)
    print(rouge_util.get_rouge())

