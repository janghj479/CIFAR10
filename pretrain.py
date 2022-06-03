""" Python Deeplearning Pytorch 참고 """
""" ResNet """
#%% 1. Module Import

import os
import numpy as np 
import matplotlib.pyplot as plt

import torch 
import torch.nn as nn 
import torch.nn.functional as F
from torchvision import transforms, datasets, models

from torch.optim import lr_scheduler


np.random.seed(42) # 같은 결과값을 위한 시드 고정
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
DIR = "D:/Git/checkpoint/"


best_acc = 0

#%% 4. Data Augmentation이 적용된 CIFAR10 데이터 다운로드(Train, Test 분리하기)

# 학습데이터에 이용하는 전처리 과정은 검증데이터에도 동일하게 적용돼야 모델의 성능을 평가할 수 있다. 

train_dataset = datasets.CIFAR10(root = "../data/CIFAR_10",
                                 train=True, 
                                 download=True, 
                                 transform = transforms.Compose([
                                     transforms.RandomCrop(32, padding=4),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])) # github 참고
                                    # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) pytorch 공식사이트
                                    # transforms.RandomResizedCrop(224)

test_dataset = datasets.CIFAR10(root = "../data/CIFAR_10",
                                 train=False, 
                                 download=True,
                                 transform =  transforms.Compose([
                                     transforms.RandomCrop(32, padding=4),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),]))


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
    
    
    
#%% 6. ResNet 모델 설계하기 
#net = models.ResNet34(pretrained=True)
net = models.resnet18(pretrained=True)
num_ftrs = net.fc.in_features
net.fc = nn.Linear(num_ftrs, 10)
    
#%% 7. Optimizer, Objective Fuction 설정하기
model = net.to(DEVICE) 
optimizer = torch.optim.Adam(model.parameters(), lr = 0.001) 
criterion = nn.CrossEntropyLoss() 

exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

print(model)

#from torchsummary import summary
#summary(model, (3, 32, 32)) 

def my_plot(Epoch, loss):
    plt.plot(Epoch, loss)

#%% 8. 모델 학습 & 학습 데이터에 대한 모델 성능 확인하는 함수 정의
def train(model, train_loader, optimizer, scheduler, log_interval): # scheduler 추가
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
eval_loss=[]
eval_acc=[]

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
    
    eval_loss.append(test_loss)
    eval_acc.append(test_accuracy)
    
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
      #best_acc = 0
    

    return test_loss, test_accuracy

#%% 10. 체크포인트 정의 

def save_checkpoint(directory, state, filename='latest.tar.gz'):

    if not os.path.exists(directory):
        os.makedirs(directory)

    model_filename = os.path.join(directory, filename)
    torch.save(state, model_filename)
    print("=> saving checkpoint")

def load_checkpoint(directory, filename='latest.tar.gz'):

    model_filename = os.path.join(directory, filename)
    if os.path.exists(model_filename):
        print("=> loading checkpoint")
        state = torch.load(model_filename)
        return state
    else:
        return None


#%% 11. 모델 학습을 진행하며 Train, Test set의 loss 및 accuracy 확인

start_epoch = 0

"""
checkpoint = load_checkpoint(DIR)

if not checkpoint:
    pass
else:
    start_epoch = checkpoint['Epoch'] + 1
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
"""

for Epoch in range(start_epoch, EPOCHS + 1):

    train(model, train_loader, optimizer, exp_lr_scheduler, log_interval = 200) 
    save_checkpoint(DIR, {
        'Epoch': Epoch,
        'model': model,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    })
    test_loss, test_accuracy = evaluate(model, test_loader) 
    
    print("\n[EPOCH: {}], \tTest Loss: {:.4f}, \tTest Accuracy: {:.2f} % \n".format(
        Epoch, test_loss, test_accuracy))
    
    if Epoch == EPOCHS:
        from matplotlib.offsetbox import AnchoredText
        fig, ax1 = plt.subplots()

        plt.grid()

        #%.2f
        textstr = '\n'.join((
            r'best_acc(epoch)=%.2f(%d)' %(best_acc, Epoch),
            r'EPOCH=%d' %(EPOCHS),
            r'BATCH_SIZE=%d' %(BATCH_SIZE)))

        textbox = AnchoredText(textstr, loc='lower right')
        ax1.add_artist(textbox)

        ax1.plot(eval_loss, alpha=0.5, color='green')
        line1 = ax1.plot(eval_loss, color='green', label='loss')

        ax2 = ax1.twinx()
        ax2.plot(eval_acc, alpha=0.5, color='deeppink')
        line2 = ax2.plot(eval_acc, color='deeppink', label='acc')

        #plt.legend(['Train','Valid'])
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='upper left')


        plt.title('Train vs Valid Accuracy')

        plt.show()      

#%%
