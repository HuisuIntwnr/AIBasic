import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from torch import optim
from torch.optim.lr_scheduler import StepLR

from torchvision import datasets
import torchvision.transforms as transforms
import torchvision
from torch.utils.data import DataLoader
import os

from torchvision import utils
import matplotlib.pyplot as plt

import numpy as np
from torchsummary import summary
import time
import copy

from torch.optim.lr_scheduler import ReduceLROnPlateau


#* ---------------------------------------------- data 
path_data = './AIBasic/resnet/data'

trainset = datasets.STL10(path_data, split='train', download=True, transform=transforms.ToTensor())
valset = datasets.STL10(path_data, split='test', download=True, transform=transforms.ToTensor())

classes = ['airplane', 'bird', 'car', 'cat', 'deer', 'dog', 'horse', 'monkey', 'ship', 'truck']

#* ---------------------------------------------- hyper-parameter
batch_size = 32

print(f'trainset shape: {trainset.data.shape} / valset shape: {valset.data.shape}')

#* ---------------------------------------------- normalize 하기위한 계산
'''
train_meanRGB = [np.mean(x.numpy(), axis=(1,2)) for x,_ in trainset]    # 이미지 각각의 RGB 평균 -> 5000장 있음
train_stdRGB = [np.std(x.numpy(), axis=(1,2)) for x,_ in trainset]

train_meanR = np.mean([m[0] for m in train_meanRGB])
train_meanG = np.mean([m[1] for m in train_meanRGB])
train_meanB = np.mean([m[2] for m in train_meanRGB])
train_stdR = np.mean([s[0] for s in train_stdRGB])
train_stdG = np.mean([s[1] for s in train_stdRGB])
train_stdB = np.mean([s[2] for s in train_stdRGB])


val_meanRGB = [np.mean(x.numpy(), axis=(1,2)) for x, _ in valset]
val_stdRGB = [np.std(x.numpy(), axis=(1,2)) for x, _ in valset]

val_meanR = np.mean([m[0] for m in val_meanRGB])
val_meanG = np.mean([m[1] for m in val_meanRGB])
val_meanB = np.mean([m[2] for m in val_meanRGB])

val_stdR = np.mean([s[0] for s in val_stdRGB])
val_stdG = np.mean([s[1] for s in val_stdRGB])
val_stdB = np.mean([s[2] for s in val_stdRGB])

print('trainset mean of R,G,B:', train_meanR, train_meanG, train_meanB)
print('trainset std of R,G,B:', train_stdR, train_stdG, train_stdB)

print('valset mean of R,G,B:', val_meanR, val_meanG, val_meanB)
print('valset std of R,G,B:', val_stdR, val_stdG, val_stdB)
'''
#* ---------------------------------------------- image transformation
train_transformation = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(224),
    transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5]),            #([train_meanR, train_meanG, train_meanB], [train_stdR, train_stdG, train_stdB]),
    transforms.RandomHorizontalFlip()
])
val_transformation = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(224),
    transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])            #([val_meanR, val_meanG, val_meanB], [val_stdR, val_stdG, val_stdB])
])

#* ---------------------------------------------- transformed data
trainset = datasets.STL10(path_data, download=False, transform=train_transformation)
valset = datasets.STL10(path_data, download=False, transform=val_transformation)

#* ---------------------------------------------- data loader
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True) 
valloader = DataLoader(valset, batch_size=batch_size, shuffle=True, drop_last=True) 

#* ---------------------------------------------- display sample images 
def _imshow(img, y=None, color=True):
    img = img / 2 + 0.5
    npimg = img.numpy()
    npimg_tr = np.transpose(npimg, (1,2,0))
    plt.title('A batch')
    plt.imshow(npimg_tr)
    plt.show()

    if y is not None:
        plt.title('labels :' + str(y))

dataiter = iter(trainloader)
images, labels = next(dataiter)

print('image size:', images.shape)
_imshow(torchvision.utils.make_grid(images))
print('Class: ', [classes[i] for i in labels])

#* ---------------------------------------------- residual block
class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLu(),
            nn.Conv2d(out_channels, out_channels * BasicBlock.expansion, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels * BasicBlock.expansion)
        )

        self.shortcut = nn.Sequential()

        self.relu = nn.ReLu()

        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion)
            )
    
    def forward(self, x):
        x = self.residual_function(x) + self.shortcut(x)
        x = self.relu(x)
        return x
    
class BottleNeck(nn.Module):
    expansion = 4
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size=1,stride=1,bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels,out_channels,kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels*BottleNeck.expansion, kernel_size=1,stride=1,bias=False),
            nn.BatchNorm2d(out_channels*BottleNeck.expansion)
        )

        self.shortcut = nn.Sequential()
        self.relu = nn.ReLU()

        if stride != 1 or in_channels != out_channels * BottleNeck.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels*BottleNeck.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels*BottleNeck.expansion)
            )

    def forward(self, x):
        x = self.residual_function(x) + self.shortcut(x)
        x = self.relu(x)
        return x

class ResNet(nn.Module):
    def __init__(self, block, num_block, num_classes=10, init_weights=True):
        super().__init__()

        self.in_channels=64

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        )

        self.conv2_x = self._make_layer(block, 64, num_block[0], 1)
        self.conv3_x = self._make_layer(block, 128, num_block[1], 2)
        self.conv4_x = self._make_layer(block, 256, num_block[2], 2)
        self.conv5_x = self._make_layer(block, 512, num_block[3], 2)

        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512*block.expansion, num_classes)

        if init_weights:
            self._initialize_weights()

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)
    
    def forward(self,x):
        output = self.conv1(x)
        output = self.conv2_x(output)
        x = self.conv3_x(output)
        x = self.conv4_x(x)
        x = self.conv5_x(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight,1)
                nn.init.constant_(m.bias,0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight,0,0.01)
                nn.init.constant_(m.bias,0)

def resnet18():
    return ResNet(BasicBlock, [2,2,2,2])

def resnet34():
    return ResNet(BasicBlock, [3,4,6,3])

def resnet50():
    return ResNet(BottleNeck, [3,4,6,3])

def resnet101():
    return ResNet(BottleNeck, [3,4,23,3])

def resnet152():
    return ResNet(BottleNeck, [3,8,36,3])

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Used device is', device)
model = resnet50().to(device)
x = torch.randn(3, 3, 224, 224).to(device)
output = model(x)
print(output.size())

summary(model, (3, 224, 224), device=device.type)

#* ---------------------------------------------- definition for model train
criterion = nn.CrossEntropyLoss(reduction='sum')
optimizer = optim.Adam(model.parameters(), lr=0.001)

lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)
    

def metric_batch(output, target):
    pred = output.argmax(1, keepdim=True)
    corrects = pred.eq(target.view_as(pred)).sum().item()
    return corrects

def loss_batch(loss_func, output, target, opt=None):            # batch마다 error 계산
    loss = loss_func(output, target)
    metric_b = metric_batch(output, target)

    if opt is not None:             # optimizer
        opt.zero_grad()
        loss.backward()
        opt.step()

    return loss.item(), metric_b

def loss_epoch(model, loss_func, dataset_dl, sanity_check=False, opt=None): # epoch의 error 계산하여 통계에 반영(학습에 안쓰임)
    running_loss = 0.0
    running_metric = 0.0
    len_data = len(dataset_dl.dataset)

    for xb, yb in dataset_dl:
        xb = xb.to(device)
        yb = yb.to(device)
        output = model(xb)

        loss_b, metric_b = loss_batch(loss_func, output, yb, opt)

        running_loss += loss_b
        
        if metric_b is not None:
            running_metric += metric_b
        
        if sanity_check is True:
            break

    loss = running_loss / len_data
    metric = running_metric / len_data

    return loss, metric

def train_val(model, params):
    num_epochs=params['num_epochs']
    loss_func=params["loss_func"]
    opt=params["optimizer"]
    train_dl=params["train_dl"]
    val_dl=params["val_dl"]
    sanity_check=params["sanity_check"]
    lr_scheduler=params["lr_scheduler"]
    path2weights=params["path2weights"]

    loss_history = {'train': [], 'val': []}
    metric_history = {'train': [], 'val': []}

    # # GPU out of memoty error
    # best_model_wts = copy.deepcopy(model.state_dict())

    best_loss = float('inf')

    start_time = time.time()

    for epoch in range(1, num_epochs+1):
        print('Epoch {}/{}'.format(epoch, num_epochs+1))

        model.train()
        train_loss, train_metric = loss_epoch(model, loss_func, train_dl, sanity_check, opt)
        loss_history['train'].append(train_loss)
        metric_history['train'].append(train_metric)

        model.eval()
        with torch.no_grad():
            val_loss, val_metric = loss_epoch(model, loss_func, val_dl, sanity_check)
        loss_history['val'].append(val_loss)
        metric_history['val'].append(val_metric)

        if val_loss < best_loss:
            best_loss = val_loss
            # best_model_wts = copy.deepcopy(model.state_dict())

            # torch.save(model.state_dict(), path2weights)
            # print('Copied best model weights!')
            print('Get best val_loss')

        lr_scheduler.step(val_loss)

        print('train loss: %.6f, val loss: %.6f, accuracy: %.2f, time: %.4f min' %(train_loss, val_loss, 100*val_metric, (time.time()-start_time)/60))
        print('-'*10)

    # model.load_state_dict(best_model_wts)

    return model, loss_history, metric_history

#* ---------------------------------------------- train start
params_train = {
    'num_epochs':1,
    'optimizer':optimizer,
    'loss_func':criterion,
    'train_dl':trainloader,
    'val_dl':valloader,
    'sanity_check':False,
    'lr_scheduler':lr_scheduler,
    'path2weights':'./models/stl10_resnet.pt',
}

model, loss_hist, metric_hist = train_val(model, params_train)

#* ---------------------------------------------- 결과 출력
correct = 0
total = 0
correct_class = {classname: 0 for classname in classes}      # dictonary
total_class = {classname: 0 for classname in classes}

with torch.no_grad():
    for data in valloader:
        images, labels = data[0].to(device), data[1].to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs,1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        for i in range(batch_size):               # drop_last
            if (list(labels)[i].item() == list(predicted)[i].item()):
                correct_class[classes[list(labels)[i].item()]]+=1
            total_class[classes[list(labels)[i].item()]]+=1


for i in range(len(classes)):
    accuracy = correct_class[classes[i]] / total_class[classes[i]] * 100
    print(f'Accuracy of {classes[i]}: {accuracy}%')
          
print(f'Accuracy of the network on the {total} val images: {correct/total*100:.1f}%')
