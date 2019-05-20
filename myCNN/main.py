import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from model import CNN
import numpy as np
import random

train_dataset = dsets.MNIST(root = './MNIST',
                            train = True,
                            transform = transforms.ToTensor(),
                            download=True)


test_dataset = dsets.MNIST(root = './MNIST',
                           train=False,
                           transform=transforms.ToTensor())

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

def data_loader(batch_size = 100):
    
    train_loader = torch.utils.data.DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle = True)
    test_loader = torch.utils.data.DataLoader(dataset = test_dataset, batch_size = batch_size,)

    return train_loader, test_loader


def main():

    setup_seed(2020)

    train_loader, test_loader = data_loader()

    epoch_num = 20
            
    model = CNN().cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr = 0.01)



    count = 0
    for epoch in range(epoch_num):
        for i, (image, label) in enumerate(train_loader):
            image = Variable(image.cuda())
            label = Variable(label.cuda())
            optimizer.zero_grad()
            outputs = model(image)
            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()
            count += 1
            if count % 500==0:
                correct = 0
                total = 0
                for images,lables in test_loader:
                    images = Variable(images).cuda()
                    outputs = model(images)
                    _, pred = torch.max(outputs.data,1)
                    total += lables.size(0)
                    correct += (pred.cpu()==lables.cpu()).sum()

                accuracy = 100*float(correct)/total
                print(count, accuracy, round(loss.item(),3))
                


if __name__ == '__main__':
    main()


