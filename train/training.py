#about torch...
import torch
import torch.nn as nn
import torch
import torch.optim as optim

from PIL import Image

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

import config
import init_data

device = 'cuda' if torch.cuda.is_available() else 'cpu'

torch.manual_seed(1234)
if device =='cuda':
    torch.cuda.manual_seed_all(1234)

class dataset(torch.utils.data.Dataset):
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
        
        # label = img_path.split('\\')[-1].split('.')[0]
        label = os.path.basename(img_path).split('.')[0]
        if label == 'dog':
            label=1
        else :
            label=0
            
        return img_transformed,label

class Cnn(nn.Module):
    def __init__(self):
        super(Cnn,self).__init__()
        
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=0, stride=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=0, stride=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)
            )
        
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=0, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc1 = nn.Linear(64, 10)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(10, 2)
        self.relu = nn.ReLU()
        
    def forward(self,x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.relu(self.fc1(out))
        out = self.fc2(out)
        return out

import matplotlib.pyplot as plt
def plot_training(loss_img_save, acc_img_save, epochs, train_losses, val_losses, train_accuracies, val_accuracies):
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

def main():
    # load data from npy
    train_list = np.load('./npy/train_list.npy')
    val_list = np.load('./npy/val_list.npy')
    test_list = np.load('./npy/test_list.npy')

    train_data = dataset(train_list, transform=init_data.train_transforms)
    val_data = dataset(val_list, transform=init_data.val_transforms)
    test_data = dataset(test_list, transform=init_data.test_transforms)

    train_loader = torch.utils.data.DataLoader(dataset = train_data, batch_size=config.batch_size, shuffle=True )
    val_loader = torch.utils.data.DataLoader(dataset = val_data, batch_size=config.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset = test_data, batch_size=config.batch_size, shuffle=True)

    model = Cnn().to(device)
    model.train()

    optimizer = optim.Adam(params = model.parameters(),lr=config.lr)
    criterion = nn.CrossEntropyLoss()

    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []

    for epoch in range(config.epochs):
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

    plot_training(config.loss_img_save, config.acc_img_save, config.epochs, train_losses, val_losses, train_accuracies, val_accuracies)

    torch.save(model.state_dict(), config.model_name)

    total_samples = 0
    correct_predictions = 0
    total_loss = 0

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

if __name__ == "__main__":
    main()