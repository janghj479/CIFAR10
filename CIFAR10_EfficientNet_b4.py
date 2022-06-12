""" https://github.com/JosephRynkiewicz/CIFAR10/blob/master/main.py"""

import os
import numpy as np 
import matplotlib.pyplot as plt

import torch 
import torch.nn as nn 
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn.functional as F

from torchvision import transforms, datasets, models

from efficientnet_pytorch import EfficientNet


torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
np.random.seed(42) 

#%% 2. 
if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
else:
    DEVICE = torch.device('cpu')
    


print('Using PyTorch version: ', torch.__version__, 'Device: ', DEVICE)

#%% 3. 
EPOCHS = 30

batch = 32
batch_split = 1
micro_batch_size = batch // batch_split 

BATCH_SIZE = micro_batch_size


best_acc = 0

#%% 4. 

train_dataset = datasets.CIFAR10(root = "../data/CIFAR_10",
                                 train=True, 
                                 download=True, 
                                 transform = transforms.Compose([
                                     transforms.RandomResizedCrop(size=160, scale=(0.6,1.0)), 
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]))
                                    
                                    

test_dataset = datasets.CIFAR10(root = "../data/CIFAR_10",
                                 train=False, 
                                 download=True,
                                 transform =  transforms.Compose([
                                     transforms.Resize(200),
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]))


train_loader = torch.utils.data.DataLoader(dataset = train_dataset,
                                           batch_size=micro_batch_size,
                                           shuffle=True) 

test_loader = torch.utils.data.DataLoader(dataset = test_dataset,
                                           batch_size=10,
                                           shuffle=False) 


def mixup_data(x,y, alpha=1.0, lam=1.0, count=0):
    if count == 0:
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1.0
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(DEVICE)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
    
                                               

#%% 5. 
for (X_train, y_train) in train_loader:
    print("X_train: ", X_train.size(), "type: ", X_train.type())
    print("y_train: ", y_train.size(), "type: ", y_train.type())
    break

pltsize = 1
plt.figure(figsize=(10 * pltsize, pltsize))

for i in range(10):
    plt.subplot(1, 10, i+1)
    plt.axis('off') # axis 
    plt.imshow(np.transpose(X_train[i], (1, 2, 0)))
    plt.title("class: " + str(y_train[i].item()))
    
    
    
#%% 6. ResNet 
net = EfficientNet.from_pretrained('efficientnet-b4', num_classes=10)
    
#%% 7. Optimizer, Objective Fuction 
model = net.to(DEVICE) 
optimizer = optim.SGD(net.parameters(), lr = 0.01, momentum=0.9, weight_decay=1e-6)
lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)  
criterion = nn.CrossEntropyLoss() 

print(model)


def my_plot(Epoch, loss):
    plt.plot(Epoch, loss)

#%% 8. 
def train(model, train_loader, optimizer, log_interval): # scheduler 
    count = 0
    lam = 1.0
    alpha = 0.1
    model.train() 
    optimizer.zero_grad()
    
    for batch_idx, (image, label) in enumerate(train_loader): 
        
        if count == batch_split:
            optimizer.step()
            optimizer.zero_grad()
            count = 0 
        
        image, label = image.to(DEVICE), label.to(DEVICE) 
        image, labels_a, labels_b, lam = mixup_data(image, label, alpha, lam, count)
        output = model(image) 
         
        loss = mixup_criterion(criterion, output, labels_a, labels_b, lam) 
        loss = loss / batch_split 
        loss.backward() 
        count += 1
        optimizer.step() 
        
        
        if batch_idx % log_interval == 0: 
            print("Train Epoch: {} [{}/{}({:.0f}%)]\tTrain Loss: {:.6f}".format(
                Epoch, batch_idx * len(image),
                len(train_loader.dataset), 100. * batch_idx / len(train_loader),
                loss.item()))
    
    
        

#%% 9. 
eval_loss=[]
eval_acc=[]

def evaluate(model, test_loader): 
    
    global best_acc
    model.eval()
    test_loss = 0 
    correct =0
    with torch.no_grad(): 
        for image, label in test_loader: 
            image, label = image.to(DEVICE), label.to(DEVICE)
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
        #best save high test accuracy save
        # and best save
        
     
    if Epoch%9 == 0:
        print('{}'.format(Epoch) + 'th epoch current best accuracy')
        print(best_acc)
    if Epoch == EPOCHS:
      print(best_acc)
      #best_acc = 0
    

    return test_loss, test_accuracy


#%% 11. 

start_epoch = 0

for Epoch in range(start_epoch, EPOCHS + 1):

    train(model, train_loader, optimizer, log_interval = 200) 
    lr_scheduler.step()

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
