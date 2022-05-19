""" Python Deeplearning Pytorch 참고 """
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
    
print('Using PyTorch version: ', torch.__version__, 'Device: ', DEVICE)

#%% 3. Parameter 정의
BATCH_SIZE = 32
EPOCHS = 10

#%% 4. CIFAR10 데이터 다운로드(Train, Test 분리하기)
train_dataset = datasets.CIFAR10(root = "../data/CIFAR_10",
                                 train=True,
                                 download=True,
                                 transform = transforms.ToTensor())

test_dataset = datasets.CIFAR10(root = "../data/CIFAR_10",
                                 train=False,
                                 download=True,
                                 transform = transforms.ToTensor())

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
    
# 데이터 다운로드 후 mini-batch 단위로 할당한 데이터 개수와 형태 확인
pltsize = 1
plt.figure(figuresize=(10 * pltsize, pltsize))
for i in range(10):
    plt.subplot(1, 10, i+1)
    plt.axis('off') # axis 제거
    plt.imshow(np.transpose(X_train[i]), (1, 2, 0))
    plt.title("class: ", + str(y_train[i].item()))
    
    
    
#%% 6. 