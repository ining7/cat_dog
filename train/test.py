#about torch...
import torch
import torch.nn as nn
import torch
import torch.optim as optim
import torch.nn.functional as F

from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, Dataset

#using numpy
import numpy as np

#for data load or save
import pandas as pd

#visualize some datasets
import matplotlib.pyplot as plt

#check our work directory
import os

#to unzip datasets
import zipfile

lr = 0.01 # learning_rate
batch_size = 100 # we will use mini-batch method
epochs = 20 # How much to train a model
loss_img_save = '../res/50_100_20/loss.png'
acc_img_save = '../res/50_100_20/acc.png'
model_name = '../check_point/50_100_20.pth'

device = 'cuda' if torch.cuda.is_available() else 'cpu'

torch.manual_seed(1234)
if device =='cuda':
    torch.cuda.manual_seed_all(1234)

os.listdir('../input/dogs-vs-cats-redux-kernels-edition')

os.listdir('../input/dogs-vs-cats-redux-kernels-edition')

base_dir = '../input/dogs-vs-cats-redux-kernels-edition'
train_dir = '../data/train'
test_dir = '../data/test'

with zipfile.ZipFile(os.path.join(base_dir, 'train.zip')) as train_zip:
    train_zip.extractall('../data')
    
with zipfile.ZipFile(os.path.join(base_dir, 'test.zip')) as test_zip:
    test_zip.extractall('../data')

import glob

train_list = glob.glob(os.path.join(train_dir,'*.jpg'))
test_list = glob.glob(os.path.join(test_dir, '*.jpg'))

from PIL import Image
# random_idx = np.random.randint(1,25000,size=10)

# fig = plt.figure()
# i=1
# for idx in random_idx:
#     ax = fig.add_subplot(2,5,i)
#     img = Image.open(train_list[idx])
#     plt.imshow(img)
#     i+=1

# plt.axis('off')
# plt.show()

from sklearn.model_selection import train_test_split
train_list, val_list = train_test_split(train_list, test_size=0.2)

#data Augumentation
train_transforms =  transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])


test_transforms = transforms.Compose([   
    transforms.Resize((224, 224)),
     transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
    ])

class dataset(torch.utils.data.Dataset):
    #가져와서 처리
    def __init__(self,file_list,transform=None):
        self.file_list = file_list
        self.transform = transform
        
        
    #dataset length
    def __len__(self):
        self.filelength = len(self.file_list)
        return self.filelength
    
    #load an one of images
    def __getitem__(self,idx):
        img_path = self.file_list[idx]
        img = Image.open(img_path)
        img_transformed = self.transform(img)
        
        label = img_path.split('\\')[-1].split('.')[0]
        if label == 'dog':
            label=1
        else :
            label=0
            
        return img_transformed,label

train_data = dataset(train_list, transform=train_transforms)
test_data = dataset(test_list, transform=test_transforms)
val_data = dataset(val_list, transform=test_transforms)

train_loader = torch.utils.data.DataLoader(dataset = train_data, batch_size=batch_size, shuffle=True )

test_loader = torch.utils.data.DataLoader(dataset = test_data, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(dataset = val_data, batch_size=batch_size, shuffle=True)

class Cnn(nn.Module):
    def __init__(self):
        super(Cnn,self).__init__()
        
        self.layer1 = nn.Sequential(
            nn.Conv2d(3,16,kernel_size=3, padding=0,stride=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(16,32, kernel_size=3, padding=0, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)
            )
        
        self.layer3 = nn.Sequential(
            nn.Conv2d(32,64, kernel_size=3, padding=0, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        
        self.fc1 = nn.Linear(3*3*64,10)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(10,2)
        self.relu = nn.ReLU()
        
        
    def forward(self,x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.view(out.size(0),-1)
        out = self.relu(self.fc1(out))
        out = self.fc2(out)
        return out
    #     self.layer1 = nn.Sequential(
    #         nn.Conv2d(3, 16, kernel_size=3, padding=1),
    #         nn.BatchNorm2d(16),
    #         nn.ReLU(),
    #         nn.MaxPool2d(2)
    #     )

    #     self.layer2 = nn.Sequential(
    #         nn.Conv2d(16, 32, kernel_size=3, padding=1),
    #         nn.BatchNorm2d(32),
    #         nn.ReLU(),
    #         nn.MaxPool2d(2)
    #     )

    #     self.layer3 = nn.Sequential(
    #         nn.Conv2d(32, 64, kernel_size=3, padding=1),
    #         nn.BatchNorm2d(64),
    #         nn.ReLU(),
    #         nn.MaxPool2d(2)
    #     )

    #     self.fc1 = nn.Linear(64 * 6 * 6, 10)  # 根据新的图像大小进行调整
    #     self.dropout = nn.Dropout(0.5)
    #     self.fc2 = nn.Linear(10, 2)
    #     self.relu = nn.ReLU()

    # def forward(self, x):
    #     out = self.layer1(x)
    #     out = self.layer2(out)
    #     out = self.layer3(out)
    #     out = out.view(out.size(0), -1)
    #     out = self.relu(self.fc1(out))
    #     out = self.fc2(out)
    #     return out


model = Cnn().to(device)
model.train()

optimizer = optim.Adam(params = model.parameters(),lr=lr)
criterion = nn.CrossEntropyLoss()

train_losses = []
train_accuracies = []
val_losses = []
val_accuracies = []

for epoch in range(epochs):
    epoch_loss = 0
    epoch_accuracy = 0
    
    for data, label in train_loader:
        data = data.to(device)
        label = label.to(device)
        
        output = model(data)
        loss = criterion(output, label)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        acc = ((output.argmax(dim=1) == label).float().mean())
        epoch_accuracy += acc/len(train_loader)
        epoch_loss += loss/len(train_loader)
        
    print('Epoch : {}, train accuracy : {}, train loss : {}'.format(epoch+1, epoch_accuracy,epoch_loss))
    train_losses.append(epoch_loss.item())
    train_accuracies.append(epoch_accuracy.item())
    
    with torch.no_grad():
        epoch_val_accuracy=0
        epoch_val_loss =0
        for data, label in val_loader:
            data = data.to(device)
            label = label.to(device)
            
            val_output = model(data)
            val_loss = criterion(val_output,label)
            
            
            acc = ((val_output.argmax(dim=1) == label).float().mean())
            epoch_val_accuracy += acc/ len(val_loader)
            epoch_val_loss += val_loss/ len(val_loader)
            
        print('Epoch : {}, val_accuracy : {}, val_loss : {}'.format(epoch+1, epoch_val_accuracy,epoch_val_loss))
        val_losses.append(epoch_val_loss.item())
        val_accuracies.append(epoch_val_accuracy.item())

torch.save(model.state_dict(), model_name)

import matplotlib.pyplot as plt

# Plot training and validation loss
epoch_range = range(1, epochs+1)
plt.plot(epoch_range, train_losses)
plt.plot(epoch_range, val_losses)
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper right')
plt.savefig(loss_img_save)
plt.clf()

# Plot training and validation accuracy
plt.plot(epoch_range, train_accuracies)
plt.plot(epoch_range, val_accuracies)
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper right')
plt.savefig(acc_img_save)

total_samples = 0
correct_predictions = 0
total_loss = 0
total_inference_time = 0

model.eval()

with torch.no_grad():
    for data, labels in test_loader:
        data = data.to(device)
        labels = labels.to(device)

        preds = model(data)
        _, predicted = torch.max(preds.data, 1)

        total_samples += labels.size(0)
        correct_predictions += (predicted == labels).sum().item()

        loss = criterion(preds, labels)
        total_loss += loss.item()

accuracy = correct_predictions / total_samples
average_loss = total_loss / len(test_loader)
print(f' === Test Accuracy: {accuracy:.4f}')
print(f' === Test Loss: {average_loss:.4f}')
# dog_probs = []
# model.eval()
# with torch.no_grad():
#     for data, fileid in test_loader:
#         data = data.to(device)
#         preds = model(data)
#         preds_list = F.softmax(preds, dim=1)[:, 1].tolist()
#         dog_probs += list(zip(list(fileid), preds_list))

# dog_probs.sort(key = lambda x : int(x[0]))
# dog_probs

# idx = list(map(lambda x: x[0],dog_probs))
# prob = list(map(lambda x: x[1],dog_probs))

# submission = pd.DataFrame({'id':idx,'label':prob})

# print(submission)

# submission.to_csv('result.csv',index=False)

