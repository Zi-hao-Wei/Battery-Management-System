import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import torchvision

class Resnet(nn.Module):
    def __init__(self):
        super(Resnet, self).__init__()
        self.m1=nn.BatchNorm2d(16,affine=True)
        self.m2=nn.BatchNorm2d(8,affine=True)
        self.m3=nn.BatchNorm2d(3,affine=True)
        self.conv1 = nn.Conv2d(18, 16, kernel_size=3)
        self.conv2 = nn.Conv2d(16, 8, kernel_size=3)
        self.conv3 = nn.Conv2d(8, 3, kernel_size=1)
        self.resnet = torchvision.models.resnet50(pretrained=False)
        # inplanes=self.resnet.inplanes
        # self.resnet.conv1 = nn.Conv2d(15, inplanes, kernel_size=7, stride=2, padding=3,
        #                        bias=False)
        fc_inputs = self.resnet.fc.in_features
        self.resnet.fc= nn.Sequential(
            nn.Linear(fc_inputs, 128),
            nn.ReLU(),
            # nn.Dropout(0.2),

            nn.Linear(128, 32),
            nn.ReLU(),
            # nn.Dropout(0.2),

            nn.Linear(32,16),
            nn.ReLU(),
            # nn.Dropout(0.1),

            nn.Linear(16,1)
        )
    def forward(self, x):
        x= self.conv1(x)
        x=self.m1(x)
        x= self.conv2(x)
        x=self.m2(x)
        x= self.conv3(x)
        x=self.m3(x)
 
        x= self.resnet(x)
        return x