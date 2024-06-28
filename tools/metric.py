from sklearn.metrics import accuracy_score


class Metric:
    def __init__(self) -> None:
        self.acc=0
        
        #acc
        self.right_nums=0
        self.total_nums=0

        #ppl

    def update_acc(self,pred,target):
        pred=pred.cpu().reshape(-1).list()
        target=target.cpu().reshape(-1).list()

        self.right_nums+=accuracy_score(pred,target,normalize=False)
        self.total_nums+=len(pred)

    def get_acc(self):
        return self.right_nums/self.total_nums

    def get_ppl(self,pred,target):
        target_list=target.reshape(-1).list()


        for right_index in target_list:
            if right_index != 0:
                pass