import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms


from torchvision import models 
from torchvision.models.feature_extraction import create_feature_extractor 

import numpy as np
import time

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor()
    # transforms.Normalize((0.5),(0.5))
])

device = ('cuda' if torch.cuda.is_available() else 'cpu')

mnist_data_train = torchvision.datasets.MNIST(root='datasets/', download = True, train = True, transform=transform)
mnist_data_test = torchvision.datasets.MNIST(root='datasets/', download=True, train = False, transform = transform)

dataloader_train = DataLoader(dataset = mnist_data_train, batch_size = 64, shuffle = True)#, num_workers=0)
dataloader_test = DataLoader(dataset = mnist_data_test, batch_size = 32, shuffle = True)


class ResNet(nn.Module):
    
    def __init__(self):
        super().__init__()
        #loadthecustomeroperatorusingtheDLLfile
        self.resnet18 = models.resnet18()
        layer_keys=["layer1","layer2","layer3","layer4"]
        # createthe featureextractor
        self.feature_extractor = create_feature_extractor(self.resnet18, layer_keys)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, 10)

        self.convf = nn.Conv2d(1,3,kernel_size = 1,stride = 1,padding = 0)

        if True:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 0.01)
                    nn.init.constant_(m.bias, 0)

    def forward(self,x):
        x= self.convf(x)
        x=self.feature_extractor(x)['layer4']
        # print(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

model = ResNet().to(device)

optimizer = optim.SGD(model.parameters(), lr=0.001)
schduler = lr_scheduler.StepLR(optimizer, step_size= 5, gamma=0.1)
criterion = nn.CrossEntropyLoss()



epochs = 30
train_loss = [] 
val_loss = []
t_accuracy_gain = []
accuracy_gain = []

for epoch in range(epochs):
   
    total_train_loss = 0
    total_val_loss = 0

    model.train()
    total_t = 0
    # training our model
    for idx, (image, label) in enumerate(dataloader_train):

        image, label = image.to(device), label.to(device)
        optimizer.zero_grad()
        pred_t = model(image)

        loss = criterion(pred_t, label)
        total_train_loss += loss.item()
        
        loss.backward()
        optimizer.step()
        
        pred_t = torch.nn.functional.softmax(pred_t, dim=1)
        for i, p in enumerate(pred_t):
            if label[i] == torch.max(p.data, 0)[1]:
                total_t = total_t + 1
                
    accuracy_t = total_t / len(mnist_data_train)
    t_accuracy_gain.append(accuracy_t)



    total_train_loss = total_train_loss / (idx + 1)
    train_loss.append(total_train_loss)
    
    # validating our model
    model.eval()
    total = 0
    for idx, (image, label) in enumerate(dataloader_test):
        image, label = image.to(device), label.to(device)
        pred = model(image)
        loss = criterion(pred, label)
        total_val_loss += loss.item()

        pred = torch.nn.functional.softmax(pred, dim=1)
        for i, p in enumerate(pred):
            if label[i] == torch.max(p.data, 0)[1]:
                total = total + 1

    accuracy = total / len(mnist_data_test)
    accuracy_gain.append(accuracy)

    total_val_loss = total_val_loss / (idx + 1)
    val_loss.append(total_val_loss)

    #if epoch % 5 == 0:
    print('\nEpoch: {}/{}, Train Loss: {:.4f}, Val Loss: {:.4f}, Val Acc: {:.4f}'.format(epoch, epochs, total_train_loss, total_val_loss, accuracy))