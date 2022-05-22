""" Python Deeplearning Pytorch 참고 """
#%% 1. Module Import

import numpy as np # 선형대수 관련 함수 이용
import matplotlib.pyplot as plt # 수치 시각화

import torch # 딥러닝 프레임워크 중 하나인 Pytorch 기본 모듈
import torch.nn as nn # 신경망 모델 설계시 필요한 함수 모아놓은 모듈
import torch.nn.functional as F # 자주 이용되는 함수를 F로 지정
from torchvision import transforms, datasets # CV분야에서 자주 이용되는 torchvision 모듈 내에서 필요한 함수 임포트

#%% 2. 딥러닝 모델을 설계할 때 활용하는 장비 확인
if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
else:
    DEVICE = torch.device('cpu')
    
# GPU 사용할지 CPU 사용할지 설정    

print('Using PyTorch version: ', torch.__version__, 'Device: ', DEVICE)

#%% 3. Parameter 정의
BATCH_SIZE = 32 # Mini-batch 1개의 단위에 대해 데이터가 32개로 구성됨
EPOCHS = 10 # Mini-batch로 학습을 1회 진행할때 횟수 "iteration", 전제 데이터를 이용해 학습을 진행한 횟수 "epoch"
            # iteration => 전체 데이터 개수 / Mini-batch 만큼 진행된다. 

#%% 4. CIFAR10 데이터 다운로드(Train, Test 분리하기)
train_dataset = datasets.CIFAR10(root = "../data/CIFAR_10", # 저장될 장소 지정
                                 train=True, # 학습용 데이터셋 다운로드
                                 download=True, # 인터넷상에서 데이터를 다운로드할 것인지 결정
                                 transform = transforms.ToTensor()) # 데이터 이용시 기본적인 전처리 진행 
                                                                    # + ToTensor method 이용해 Tensor 형태로 변경

test_dataset = datasets.CIFAR10(root = "../data/CIFAR_10",
                                 train=False, # 검증용 데이터셋 다운로드
                                 download=True,
                                 transform = transforms.ToTensor())

# 다운로드한 데이터셋을 Mini-Batch 단위로 분리해 지정
# train_loader : 학습용 데이터셋 이용해 모델 학습 진행 
# test_loader : 검증용 데이터셋 이용해 모델 성능 확인

train_loader = torch.utils.data.DataLoader(dataset = train_dataset, # 학습용 데이터셋 이용
                                           batch_size=BATCH_SIZE, # Mini-Batch 단위를 구성하는 데이터 개수 지정
                                           shuffle=True) # 데이터 순서 섞어줌 => 데이터 label만 외워서 잘못된 방향으로 학습하는것 방지

test_loader = torch.utils.data.DataLoader(dataset = test_dataset, # 검증용 데이터셋 이용
                                           batch_size=BATCH_SIZE,
                                           shuffle=False) 

#%% 5. 데이터 확인하기 
for (X_train, y_train) in train_loader:
    print("X_train: ", X_train.size(), "type: ", X_train.type())
    print("y_train: ", y_train.size(), "type: ", y_train.type())
    break
    
# 데이터 다운로드 후 mini-batch 단위로 할당한 데이터 개수와 형태 확인
# X_train:  torch.Size([32, 3, 32, 32]) type:  torch.FloatTensor
# 32개의 이미지 데이터, 채널 3개(R,G,B), 가로 픽셀, 세로 픽셀 
# [Batch_Size, Channel, Height, Width]

# y_train:  torch.Size([32]) type:  torch.LongTensor
# 32개의 이미지 각각에 대해 label 하나씩 존재            

# 데이터 시각화
# 시각화하기 위해 [width, height, channel] 형태로 변환 후 시각화
pltsize = 1
plt.figure(figsize=(10 * pltsize, pltsize))
for i in range(10):
    plt.subplot(1, 10, i+1)
    plt.axis('off') # axis 제거
    plt.imshow(np.transpose(X_train[i], (1, 2, 0)))
    plt.title("class: " + str(y_train[i].item()))
    
    
    
#%% 6. MLP 모델 설계하기 

class Net(nn.Module): # nn.Module 클래스 상속받는 Net 클래스 정의
    def __init__(self): # 인스턴스 생성시 갖게 되는 성질 정의
        super(Net, self).__init__() # nn.Module 내에 있는 메서드 상속받아 이용
        self.fc1 = nn.Linear(32 * 32 * 3, 512) # FC Layer 정의 (가로 * 세로 * 채널) 크기의 노드 수 설정, 두번째 레이어 설정할 노드 수 정의
        self.fc2 = nn.Linear(512, 256) # FC Layer 정의 (input, output)
        self.fc3 = nn.Linear(256, 10) # 최종 FC Layer 정의 (input, output) 이때 output은 label 개수와 동일하게 설정
        
    def forward(self, x): # forward propagation 정의
        x = x.view(-1, 32 * 32 * 3) # input을 1차원 벡터값으로 입력하기 위해 view 메서드를 이용해 flatten
        x = self.fc1(x) # __init__ method를 이용해 정의한 1번째 fc layer에 1차원으로 펼친 이미지 데이터 통과 
        x = F.relu(x) # Non-linear한 ReLU 함수에 통과
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.log_softmax(x, dim = 1) #log.softmax를 이용해 최종 output 계산 
        # softmax가 아닌 log_softmax를 이용하는 이유 => Back Propagation을 이용해 학습이 원할하게 진행되도록 함
        return x # 최종 계산된 x값을 output로 반환
    
#%% 7. Optimizer, Objective Fuction 설정하기
model = Net().to(DEVICE) # 정의한 모델을 디바이스에 할당
optimizer = torch.optim.Adam(model.parameters(), lr = 0.001) # parameter업데이트할 때 이용되는 옵티마이저 정의
criterion = nn.CrossEntropyLoss() # MLP 모델의 output값과 계산될 label값은 데이터의 class를 표현하는 원-핫 인코딩 값이다.
                                  # MLP 모델의 output값과 원-핫 인코딩 값의 loss는 CrossEntropyLoss를 이용해 계산

print(model)

#%% 8. 모델 학습 & 학습 데이터에 대한 모델 성능 확인하는 함수 정의
def train(model, train_loader, optimizer, log_interval):
    model.train() # 정의한 모델을 학습 상태로 지정
    for batch_idx, (image, label) in enumerate(train_loader): 
        # 기존에 정의한 train_loader에는 이미지 데이터와 레이블 데이터가 mini-batch 단위로 묶여서 저장되어 있음 -> for문 이용하여 순서대로 이용
        image = image.to(DEVICE) # 기존에 정의한 장비에 데이터 할당
        label = label.to(DEVICE)
        optimizer.zero_grad() # optimizer의 gradient 초기화
        output = model(image) # image 데이터를 input으로 이용해 output 계산
        loss = criterion(output, label) #계산된 output값과 할당된 label 데이터를 기존에 정의한 crossentropy를 이용해 loss계산
        loss.backward() # Back Propagation 실행
        optimizer.step() # 파라미터 값 업데이트
        
        if batch_idx % log_interval == 0: #log_interval 학습이 진행되면서 Mini-Batch의 Index를 이용해 과정을 모니터링할 수 있도록 함.
            print("Train Epoch: {} [{}/{}({:.0f}%)]\tTrain Loss: {:.6f}".format(
                Epoch, batch_idx * len(image),
                len(train_loader.dataset), 100. * batch_idx / len(train_loader),
                loss.item()))
            

#%% 9. 검증 데이터에 대한 모델 성능 확인하는 함수 정의
def evaluate(model, test_loader): # 모델 평가상태 설정
    model.eval()
    test_loss = 0 
    correct =0
    with torch.no_grad(): # 모델 평가시 파라미터 업데이트 방지
        for image, label in test_loader: # mini-batch로 되어있는 값 접근
            image = image.to(DEVICE)
            label = label.to(DEVICE)
            output = model(image)
            test_loss += criterion(output, label).item() # 계산한 결과값 업데이트
            prediction = output.max(1, keepdim = True)[1] # 계산된 벡터값 위치에 대응하는 클래스로 예측했다고 판단 
            correct += prediction.eq(label.view_as(prediction)).sum().item() # 예측한 클래스와 실제 레이블이 동일하면 correct에 추가
    test_loss /= len(test_loader.dataset) # 현재까지 계산된 test_loss값을 test_loader 내에 존재하는 mini-batch 개수만큼 나눠 평균 loss값으로 계산
    test_accuracy = 100. * correct / len(test_loader.dataset) # test_loader 데이터 중 얼마나 맞췄는지 정확도 계산
    return test_loss, test_accuracy # 값 반환

#%% 10. 모델 학습을 진행하며 Train, Test set의 loss 및 accuracy 확인
# 모델 학습 및 검증 진행
for Epoch in range(1, EPOCHS + 1):
    train(model, train_loader, optimizer, log_interval = 200) # log_interval -> 학습이 진행되면서 mini-batch의 index 이용해 과정 모니터링
    test_loss, test_accuracy = evaluate(model, test_loader) 
    print("\n[EPOCH: {}], \tTest Loss: {:.4f}, \tTest Accuracy: {:.2f} % \n".format(
        Epoch, test_loss, test_accuracy))
    
