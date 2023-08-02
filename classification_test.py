import torchvision
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim



#* --------------------------------- 기본정의
transform = transforms.Compose(
    [transforms.ToTensor(),                                         # 0~1 범위
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]       # -1~1 범위
)
batch_size = 4
num_epoch = 2

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

#* --------------------------------- trainset sample image, label 확인
def imshow(img):
    img = img / 2 + 0.5     # unnormalize(-1~1 을 다시 0~1로) - numpy의 image는 0~1
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))  # matplotlib 형식 - (H,W,C)
    plt.show()

dataiter = iter(trainloader)
images, labels = next(dataiter)
print('image size:', images.shape)
imshow(torchvision.utils.make_grid(images))
print('Class: ', [classes[i] for i in labels])

#* --------------------------------- def for train
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3,6,5)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(6,16,5)
        self.fc1 = nn.Linear(16*5*5,60)     # 계산 필요
        self.fc2 = nn.Linear(60,10)

    def forward(self, img):
        img = self.pool(F.relu(self.conv1(img)))
        img = self.pool(F.relu(self.conv2(img)))
        img = torch.flatten(img, 1)
        img = F.relu(self.fc1(img))
        img = F.relu(self.fc2(img))
        return img


net = CNN()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
net.to(device)

#! loss 함수, optimizer 정의 및 생성
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

#* --------------------------------- train start
for epoch in range(num_epoch):

    running_loss = 0.0  
    for i , data in enumerate(trainloader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)

        optimizer.zero_grad()

        outputs = net.forward(inputs)   
        once_loss = criterion(outputs, labels)
        once_loss.backward()
        optimizer.step()

        running_loss += once_loss.item()
        if i%2000 == 1999:
            print(f'<epoch: {epoch+1}, batch: {i+1}> - loss={running_loss/2000}% ')
            running_loss = 0.0

#* --------------------------------- model save
PATH = './cifar_net.pth'
torch.save(net.state_dict(), PATH)

#* --------------------------------- testset sample image, label 확인
dataiter = iter(testloader)
images, labels = next(dataiter)
imshow(torchvision.utils.make_grid(images))
print('GroundTruth Class: ', [classes[i] for i in labels])
#* --------------------------------- trained model load
net = CNN()
net.load_state_dict(torch.load(PATH))

#* --------------------------------- sample of test
outputs = net.forward(images)
_, predicted = torch.max(outputs,1)
print('GroundTruth Class: ', [classes[i] for i in predicted])

#* --------------------------------- test start
correct = 0
total = 0
correct_class = {classname: 0 for classname in classes}      # dictonary
total_class = {classname: 0 for classname in classes}

with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs,1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        for i in range(batch_size):
            if labels(i) == predicted(i):
                correct_class[classes[labels[i]]]+=1
            total_class[classes[labels[i]]]+=1

for i in range(len(classes)):
    accuracy = correct_class[i] / total_class[i] * 100
    print(f'Accuracy of {classes[i]}: {accuracy}%')
          
print(f'Accuracy of the network on the {total} test images: {correct/total*100:.1f}%')

    



