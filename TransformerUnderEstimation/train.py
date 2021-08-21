import torch
import torch.nn as nn 
from MyResnet import Resnet
from DataLoader import Preprocess
from torch.autograd import Variable
import numpy as np 
import matplotlib.pyplot as plt 
import math
class resNetTrain():
    def __init__(self,device,rate,f=nn.MSELoss()):
        
        self.pre=Preprocess()
        self.trainLoader=self.pre.train()
        self.testLoader=self.pre.test()
        self.device=device
        self.model=Resnet().to(device)
        self.criterion=f
        self.rate=rate
        self.optimizer=torch.optim.Adam(self.model.parameters(),lr=self.rate)
            # decline=0.000001
        self.bestLoss=100000000
        # self.trainLossBest=10000000
        self.trainLoss=10000000
    def train(self,epochs):
        
        for i in range(epochs): 
            total_loss=0
            l=0

            for _,(inputs,labels) in enumerate(self.trainLoader):
                # print(inputs.shape)
                l+=inputs.shape[0]
                inputs,labels=inputs.to(self.device),labels.to(self.device) #GPU

                # print(inputs.shape)
                # print(labels.shape)

                pred = self.model(inputs)
                # print(pred.shape)

                pred=pred.squeeze()
                labels=labels.squeeze()
                
                # print(pred)
                # print(labels)


                loss = self.criterion(pred, labels)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                # print(loss.item())

                total_loss += (loss.item()*inputs.shape[0])
            print(l)
            print(i,math.sqrt(total_loss/l))
            self.trainLoss=math.sqrt(total_loss/l)
            self.test()

    def load(self,path):
        self.model=torch.load(path)
        self.optimizer=torch.optim.Adam(self.model.parameters(),lr=self.rate,weight_decay=0.0000001)
    def test(self):
        total_loss=0
        l=0
        accuracy=0 
        MYLoss=0
        for _,(inputs,labels) in enumerate(self.testLoader):
            l+=inputs.shape[0]
            inputs,labels=inputs.to(self.device),labels.to(self.device) #GPU
            pred = self.model(inputs)
            pred=pred.squeeze()
            labels=labels.squeeze()
            # print(pred,labels)

            # # MYLoss+=torch.sum(abs(pred-labels))
            # if inputs.shape[0]==1:
            #     MYLoss+=(pred-labels)*(pred-labels)
            # else:
            #     for i in (pred-labels):
            #         MYLoss+=i*i
            # if (pred[0,0]>pred[0,1] and labels[0]==0) or (pred[0,0]<pred[0,1] and labels[0]==1):
                # accuracy+=1

            loss = self.criterion(pred, labels)
            total_loss += (loss.item()*inputs.shape[0])
        # print("RMSE",math.sqrt(MYLoss/l))
        print("VAL-RMSE",math.sqrt(total_loss/l))
        if self.trainLoss<50:
            if total_loss<self.bestLoss:
                print("best",self.trainLoss,total_loss)
                torch.save(self.model,"MA504.pkl")
                self.bestLoss=total_loss

def main():
    # rate=0.00001
    device=torch.device("cuda:0")
    rate=0.0001

    Test=resNetTrain(device,rate)
    
    Test.load("MA503.pkl")
    Test.test()
    Test.train(1000)


if __name__ == '__main__':
    main()