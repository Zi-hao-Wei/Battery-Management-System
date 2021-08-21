import numpy as np 
from torch.utils.data import DataLoader,Dataset 
import torch
from FeatureExtraction import FeatureExtraction

class Preprocess():
    def __init__(self):
        self.FE=FeatureExtraction(train=True,process=False)
        self.Graph=self.FE.pipeline(process=False)
        # print(self.Life)
        self.Life=self.FE.getLifeTime()

    def train(self,n=20):

        # self.Graph=self.FE
        print(len(self.Graph))

        self.x=torch.tensor(self.Graph) 
        self.x = self.x.type(torch.FloatTensor)
        self.y=torch.tensor(self.Life) 
        self.y = self.y.type(torch.FloatTensor) 
        # print(y)
        train_loader = DataLoader(dataset=Mydataset(self.x, self.y),batch_size=32,shuffle=True)
        return train_loader

    def test(self):
        self.FE=FeatureExtraction(train=False,process=False)
        self.Graph=self.FE.pipeline(process=False)
        self.Life=self.FE.getLifeTime()
        return self.train()

        

class Mydataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        print(self.x.shape)
        self.y = y
        print(self.y.shape)

    def __getitem__(self, index):
        x1 = self.x[index]
        y1 = self.y[index]
        return x1, y1

    def __len__(self):
        return len(self.x)

if __name__=="__main__":
    pre=Preprocess()
    pre.train()
    pre.test()