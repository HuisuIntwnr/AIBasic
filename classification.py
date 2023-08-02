import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

#! 
#todo
#*
#?ddd
#//

#* Tensor 이미지로 변환하고 normalize 하기
transform = transforms.Compose(
    [transforms.ToTensor(),                                         # 0~1 범위
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])       # -1~1 범위

# hyper-parameter
batch_size = 4
num_epoch = 2

# data loader
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)              # output: images, labels에 대한 tuple쌍 generator

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

#! ----------------------------------- sample test -----------------------------------
#* 샘플이미지를 보여주기 위한 함수
def imshow(img):
    img = img / 2 + 0.5     # unnormalize(-1~1 을 다시 0~1로) - numpy의 image는 0~1
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))  # matplotlib 형식 - (H,W,C)
    plt.show()


#* 학습용 이미지를 무작위로 가져오기
dataiter = iter(trainloader)
images, labels = next(dataiter)

#* 샘플 이미지 보여주기
print('image size:', images.shape)
imshow(torchvision.utils.make_grid(images))

#* 정답(label) 출력
print(' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))


#! -------------------------------------- 모델 시작 --------------------------------------
#* NN 정의
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)       # 32(원래 사진 사이즈) -> 14 -> 5
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))        #input - conv1 - relu - pooling - output
        x = self.pool(F.relu(self.conv2(x)))        #input - conv2 - relu - pooling - output
        x = torch.flatten(x, 1)                     #배치를 제외한 모든 차원을 평탄화(flatten) -> 총 갯수는 변함없음.
        x = F.relu(self.fc1(x))                     #input - fully connected - relu - output
        x = F.relu(self.fc2(x))                     #input - fully connected - relu - output
        x = self.fc3(x)                             #input - fully connected - output
        return x


net = Net()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
net.to(device)

#* loss 함수, optimizer 정의 및 생성
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

#! -------------------------------------- train --------------------------------------
for epoch in range(num_epoch):   # 데이터셋을 수차례 반복합니다.(epoch)

    running_loss = 0.0      # 단지 사용자를 위한 통계에만 쓰임.
    for i, data in enumerate(trainloader, start=0):
        # [inputs, labels]의 목록인 data로부터 입력을 받은 후;
        inputs, labels = data[0].to(device), data[1].to(device)

        # 변화도(Gradient) 매개변수를 0으로 만들고
        optimizer.zero_grad()

        # 순전파 + 역전파 + 최적화를 한 후
        outputs = net(inputs)   #forward
        loss = criterion(outputs, labels)
        loss.backward() 
        optimizer.step()    # 역전파 단계에서 수집된 변화도로 매개변수를 조정

        # 통계를 출력합니다.
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches(iter)
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0

print('Finished Training')

#* 학습 후 모델 저장
PATH = './cifar_net.pth'
torch.save(net.state_dict(), PATH)

# test sample image load
dataiter = iter(testloader)
images, labels = next(dataiter)

# 이미지를 출력합니다.
imshow(torchvision.utils.make_grid(images))
print('GroundTruth: ', ' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))

#! -------------------------------------- test --------------------------------------
#* 학습한 모델 load
net = Net()
net.load_state_dict(torch.load(PATH))

#* 모델 forward
outputs = net(images)           # output은 각 class에 대한 확률

_, predicted = torch.max(outputs, 1)    # 행방향으로 최댓값 (_는 element, predicted는 index(:class num))
                                        # predicted는 tensor형태

print('Predicted:   ', ' '.join(f'{classes[predicted[j]]:5s}' for j in range(batch_size)))  # predicted의 index를 이용하여 predicted에 들은 값의 classes 값을 나열.
#! 전체
correct = 0
total = 0
# 학습 중이 아니므로, 출력에 대한 변화도를 계산할 필요가 없습니다
with torch.no_grad():
    for data in testloader:     #data: image 4장과 class label 4개
        images, labels = data
        # 신경망에 이미지를 통과시켜 출력을 계산합니다
        outputs = net(images)
        # 가장 높은 값(energy)를 갖는 분류(class)를 정답으로 선택하겠습니다
        _, predicted = torch.max(outputs.data, 1)           # 행방향으로 최댓값 (_는 element, predicted는 index(:class num))
        total += labels.size(0)                             # total: test image의 갯수(총 시험문제)
        correct += (predicted == labels).sum().item()       # correct: correct image의 갯수(총 맞은 문제) 
                                                            # (predicted == labels)는 tensor 형태 -> ex) tensor([False, True, False, True])

print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')

#! 각 class별
# 각 분류(class)에 대한 예측값 계산을 위해 준비
correct_pred = {classname: 0 for classname in classes}      # dictonary
total_pred = {classname: 0 for classname in classes}
# 변화도는 여전히 필요하지 않습니다
with torch.no_grad():           # test시에는 불필요한 autograd 엔진(train에 필요)을 꺼서, 메모리를 낭비하지 않아 연산속도를 올림.
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predictions = torch.max(outputs, 1)
        # 각 분류별로 올바른 예측 수를 모읍니다
        for label, prediction in zip(labels, predictions):          # labels: 정답, predictions: 예측
            if label == prediction:
                correct_pred[classes[label]] += 1                   # 예측과 정답이 같을 때, 해당 class의 label의 key 값(정답횟수) 증가. 
            total_pred[classes[label]] += 1                         # 무조건, 해당 class의 label의 key 값(문제갯수) 증가. 
                                                                    # -> test dataset의 균일성을 확인할 수 있음.(여기서는 모든 class의 사진이 1000개. 즉, 총 1000*10=10000의 데이터 셋)


# 각 분류별 정확도(accuracy)를 출력합니다
for classname, correct_count in correct_pred.items():
    accuracy = 100 * float(correct_count) / total_pred[classname]
    print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')    # 각 class마다 정확도 출력
