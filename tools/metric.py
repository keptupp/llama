from sklearn.metrics import accuracy_score
import torch
import math


class Metric:
    def __init__(self) -> None:
        self.acc=0
        
        #acc
        self.right_nums=0
        self.total_nums=0

        #ppl

    def update_acc(self,pred,target):
        target=target.reshape(-1).tolist()
        pred=torch.argmax(torch.softmax(pred.permute(0,2,1),dim=-1),dim=-1).reshape(-1).tolist()


        pred=[pred[i] for i,x in enumerate(target) if x != 0]
        target=[x for x in target if x != 0]
        

        self.right_nums+=accuracy_score(pred,target,normalize=False)
        self.total_nums+=len(pred)

    def get_acc(self):
        return self.right_nums/self.total_nums

    def get_ppl(self,pred,target):
        target=target.reshape(-1)
        pred=pred.permute(0,2,1)
        pred=torch.softmax(pred,dim=-1)
        pred=pred.reshape(-1,65000)

        total_log_prob=0
        total_tokens=0


        for index in range(target.shape[0]):
            if target[index] !=0:
                total_log_prob +=math.log(pred[index][target[index]])
                total_tokens+=1

        perplexity = math.exp(-total_log_prob / total_tokens)
        return perplexity