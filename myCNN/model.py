import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.datasets as dsets
import torchvision.transforms as transforms


# self.cnn1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2)
# self.relu1 = nn.ReLU()
# self.maxpool1 = nn.MaxPool2d(kernel_size=2)
# self.cnn2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2)
# self.relu2 = nn.ReLU()
# self.maxpool2 = nn.MaxPool2d(kernel_size=2)
# self.fcl = nn.Linear(32*7*7, 10)

class CNN(nn.Module):
    
    def __init__(self):
        super(CNN, self).__init__()
        # 28*28
        self.cnn1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5)
        self.cnn3 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=21)
        # 24*24*32
        self.batchnormal32 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        # 12*12*32
        self.cnn2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5)
        # 8*8*64
        self.batchnormal64 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        # 4*4*64
        self.fcl = nn.Linear(4*4*64, 10)
        
        
    def forward(self, x):
        out = self.cnn1(x)
        out = self.batchnormal32(out)
        out = self.relu1(out)
        out = self.maxpool1(out)
        out = self.cnn2(out)

        out2= self.cnn3(x)
        out = out + out2

        out = self.batchnormal64(out)
        out = self.relu2(out)
        out = self.maxpool2(out)
        out = out.view(out.size(0), -1)
        out = self.fcl(out)
        return out