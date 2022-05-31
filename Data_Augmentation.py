""" Python Deeplearning Pytorch 참고 """
""" CNN """
#%% 1. Module Import

import numpy as np 
import matplotlib.pyplot as plt

import torch 
import torch.nn as nn 
import torch.nn.functional as F
from torchvision import transforms, datasets

#%% 2. 딥러닝 모델을 설계할 때 활용하는 장비 확인
if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
else:
    DEVICE = torch.device('cpu')
    
# GPU 사용할지 CPU 사용할지 설정    

print('Using PyTorch version: ', torch.__version__, 'Device: ', DEVICE)

#%% 3. Parameter 정의
BATCH_SIZE = 32 
EPOCHS = 100
            
best_acc = 0

#%% 4. Data Augmentation이 적용된 CIFAR10 데이터 다운로드(Train, Test 분리하기)

# 학습데이터에 이용하는 전처리 과정은 검증데이터에도 동일하게 적용돼야 모델의 성능을 평가할 수 있다. 

train_dataset = datasets.CIFAR10(root = "../data/CIFAR_10",
                                 train=True, 
                                 download=True, 
                                 transform = transforms.Compose( [ # 불러오는 이미지 데이터에 전처리 및 Augmentation을 다양하게 적용하는 메서드
                                     transforms.RandomHorizontalFlip(), # 해당 이미지를 50% 확률로 좌우 반전
                                     transforms.ToTensor(), # 이미지를 0~1 사이 값으로 정규화하여 Tensor 형태로 반환
                                     transforms.Normalize((0.5, 0.5, 0.5),     # tensor 형태로 변환된 이미지에 대해 정규화 (평균, 표준편차 필요)
                                                                               # (R,G,B) 순서로 평균을 0.5씩 적용
                                                          (0.5, 0.5, 0.5))]))  # (R,G,B) 순서로 표준편차를 0.5씩 적용

test_dataset = datasets.CIFAR10(root = "../data/CIFAR_10",
                                 train=False, 
                                 download=True,
                                 transform = transforms.Compose( [ # 불러오는 이미지 데이터에 전처리 및 Augmentation을 다양하게 적용하는 메서드
                                     transforms.RandomHorizontalFlip(), # 해당 이미지를 50% 확률로 좌우 반전
                                     transforms.ToTensor(), # 이미지를 0~1 사이 값으로 정규화하여 Tensor 형태로 반환
                                     transforms.Normalize((0.5, 0.5, 0.5),     # tensor 형태로 변환된 이미지에 대해 정규화 (평균, 표준편차 필요)
                                                                               # (R,G,B) 순서로 평균을 0.5씩 적용
                                                          (0.5, 0.5, 0.5))]))  # (R,G,B) 순서로 표준편차를 0.5씩 적용


train_loader = torch.utils.data.DataLoader(dataset = train_dataset,
                                           batch_size=BATCH_SIZE,
                                           shuffle=True) 

test_loader = torch.utils.data.DataLoader(dataset = test_dataset,
                                           batch_size=BATCH_SIZE,
                                           shuffle=False) 

#%% 5. 데이터 확인하기 
for (X_train, y_train) in train_loader:
    print("X_train: ", X_train.size(), "type: ", X_train.type())
    print("y_train: ", y_train.size(), "type: ", y_train.type())
    break

pltsize = 1
plt.figure(figsize=(10 * pltsize, pltsize))

for i in range(10):
    plt.subplot(1, 10, i+1)
    plt.axis('off') # axis 제거
    plt.imshow(np.transpose(X_train[i], (1, 2, 0)))
    plt.title("class: " + str(y_train[i].item()))
    
    
    
#%% 6. MLP 모델 설계하기 

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
                       
        self.conv1 = nn.Conv2d(    
            in_channels = 3,       
            out_channels = 8,       
            kernel_size = 3,       
            padding = 1)           
      
        self.conv2 = nn.Conv2d(     
            in_channels = 8,       
            out_channels = 16,     
            kernel_size = 3,        
            padding = 1)
        
        self.pool = nn.MaxPool2d(  
            kernel_size = 2,       
            stride = 2)            
                      
        self.fc1 = nn.Linear(8 * 8 * 16, 64)        
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 10)         

        
    def forward(self, x):
        x =  self.conv1(x) 
        x = F.relu(x) 
        x = self.pool(x) 
        x = self.conv2(x) 
        x = F.relu(x)
        x = self.pool(x)
        
        x = x.view(-1, 8 * 8 * 16) 
        x = self.fc1(x) 
        x = F.relu(x) 
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.log_softmax(x)
        
        return x 
    
#%% 7. Optimizer, Objective Fuction 설정하기
model = CNN().to(DEVICE) 
optimizer = torch.optim.Adam(model.parameters(), lr = 0.001) 
criterion = nn.CrossEntropyLoss() 

print(model)

from torchsummary import summary
summary(model, (3, 32, 32)) 


#%% 8. 모델 학습 & 학습 데이터에 대한 모델 성능 확인하는 함수 정의
def train(model, train_loader, optimizer, log_interval):
    model.train() 
    
    for batch_idx, (image, label) in enumerate(train_loader): 
       
        image = image.to(DEVICE) 
        label = label.to(DEVICE)
        optimizer.zero_grad() 
        output = model(image) 
        loss = criterion(output, label)
        loss.backward() 
        optimizer.step() 
        
        
        if batch_idx % log_interval == 0: 
            print("Train Epoch: {} [{}/{}({:.0f}%)]\tTrain Loss: {:.6f}".format(
                Epoch, batch_idx * len(image),
                len(train_loader.dataset), 100. * batch_idx / len(train_loader),
                loss.item()))
            

#%% 9. 검증 데이터에 대한 모델 성능 확인하는 함수 정의
def evaluate(model, test_loader): 
    global best_acc
    model.eval()
    test_loss = 0 
    correct =0
    with torch.no_grad(): 
        for image, label in test_loader: 
            image = image.to(DEVICE)
            label = label.to(DEVICE)
            output = model(image)
            test_loss += criterion(output, label).item()
            prediction = output.max(1, keepdim = True)[1] 
            correct += prediction.eq(label.view_as(prediction)).sum().item() 
    test_loss /= len(test_loader.dataset) 
    test_accuracy = 100. * correct / len(test_loader.dataset) 
    
    # best_accuracy
    if test_accuracy > best_acc:
        best_acc = test_accuracy
        print('='*50) 
        print(best_acc)   
        print('='*50) 
     
    if Epoch%9 == 0:
        print('{}'.format(Epoch) + 'th epoch current best accuracy')
        print(best_acc)
    if Epoch == EPOCHS:
      print(best_acc)
      best_acc = 0
    
    
    return test_loss, test_accuracy 

#%% 10. 모델 학습을 진행하며 Train, Test set의 loss 및 accuracy 확인

for Epoch in range(1, EPOCHS + 1):
    train(model, train_loader, optimizer, log_interval = 200) 
    test_loss, test_accuracy = evaluate(model, test_loader) 
    print("\n[EPOCH: {}], \tTest Loss: {:.4f}, \tTest Accuracy: {:.2f} % \n".format(
        Epoch, test_loss, test_accuracy))
   