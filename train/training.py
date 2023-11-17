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

from tqdm import tqdm
import math

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

class Dnn(nn.Module):
    def __init__(self):
        super(Dnn,self).__init__()
        
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, padding="same", stride=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=5, padding="same", stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
        )
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, padding="same", stride=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=5, padding="same", stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
        )
        
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding="same", stride=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding="same", stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
        )

        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding="same", stride=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding="same", stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
        )

        self.layer5 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, padding="same", stride=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=5, padding="same", stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
        )
        
        self.layer6 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, padding="same", stride=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=5, padding="same", stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
        )
        
        self.layer7 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding="same", stride=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding="same", stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
        )

        self.layer8 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding="same", stride=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding="same", stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
        )
        
        self.avg_pool = nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.fc1 = nn.Linear(6272, 1024)
        self.fc2 = nn.Linear(1024, 1)
        self.fc3 = nn.Linear(16384, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self,x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avg_pool(out)

        # bilinear model:
        out2 = self.layer5(x)
        out2 = self.layer6(out2)
        out2 = self.layer7(out2)
        out2 = self.layer8(out2)
        out2 = self.avg_pool(out2)

        print(list(out.size()))
        
        out = out.view(out.size(0), 128, 49)
        out2 = out2.view(out2.size(0), 128, 49)

        print(list(out2.size()))

        out_T = torch.transpose(out2, 1, 2)
        out = torch.bmm(out, out_T) / (49)

        print(list(out.size()))

        out = out.view(out.size(0), 128 * 128)
        
        # The signed square root
        out = torch.sign(out) * torch.sqrt(torch.abs(out) + 1e-12)
        # L2 regularization
        out = torch.nn.functional.normalize(out)

        print(list(out.size()))

        # out = out.view(out.size(0), -1)
        # out = self.relu(self.fc1(out))
        out = self.sigmoid(self.fc3(out))
        return out

class Embeddings(nn.Module):
    '''
    对图像进行编码，把图片当做一个句子，把图片分割成块，每一块表示一个单词
    '''
    def __init__(self, img_size, in_channels=3):
        super(Embeddings,self).__init__()
        ##将图片分割成多少块（224/16）*（224/16）=196
        n_patches=(img_size//config.patch_size)*(img_size//config.patch_size)
        #对图片进行卷积获取图片的块，并且将每一块映射成config.hidden_size维（768）
        self.patch_embeddings=nn.Conv2d(in_channels=in_channels,
                                     out_channels=config.hidden_size,
                                     kernel_size=config.patch_size,
                                     stride=config.patch_size)
        
        #设置可学习的位置编码信息，（1,196+1,786）
        self.position_embeddings=nn.Parameter(torch.zeros(1,
                                                          n_patches+1,
                                                          config.hidden_size))
        #设置可学习的分类信息的维度
        self.classifer_token=nn.Parameter(torch.zeros(1,1,config.hidden_size))
        self.dropout=nn.Dropout(config.dropout_rate)

    def forward(self,x):
        bs=x.shape[0]
        cls_tokens=self.classifer_token.expand(bs,-1,-1)
        
        x=self.patch_embeddings(x)#（bs,768,14,14）
        x=x.flatten(2)#(bs,768,196)
        x=x.transpose(-1,-2)#(bs,196,768)
        x=torch.cat((cls_tokens,x),dim=1)#将分类信息与图片块进行拼接（bs,197,768）

        embeddings=x+self.position_embeddings#将图片块信息和对其位置信息进行相加(bs,197,768)
        embeddings=self.dropout(embeddings)
        return  embeddings

#2.构建self-Attention模块
class Attention(nn.Module):
    def __init__(self,config,vis):
        super(Attention,self).__init__()
        self.vis=vis
        self.num_attention_heads=config.num_heads #12
        self.attention_head_size = int(config.hidden_size / self.num_attention_heads)  # 768/12=64
        self.all_head_size = self.num_attention_heads * self.attention_head_size  # 12*64=768

        self.query = nn.Linear(config.hidden_size, self.all_head_size)#wm,768->768，Wq矩阵为（768,768）
        self.key = nn.Linear(config.hidden_size, self.all_head_size)#wm,768->768,Wk矩阵为（768,768）
        self.value = nn.Linear(config.hidden_size, self.all_head_size)#wm,768->768,Wv矩阵为（768,768）
        self.out = nn.Linear(config.hidden_size, config.hidden_size)  # wm,768->768
        self.attn_dropout = nn.Dropout(config.attention_dropout_rate)
        self.proj_dropout = nn.Dropout(config.attention_dropout_rate)

        self.softmax = nn.Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (
        self.num_attention_heads, self.attention_head_size)  # wm,(bs,197)+(12,64)=(bs,197,12,64)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)  # wm,(bs,12,197,64)

    def forward(self, hidden_states):
        # hidden_states为：(bs,197,768)
        mixed_query_layer = self.query(hidden_states)#wm,768->768
        mixed_key_layer = self.key(hidden_states)#wm,768->768
        mixed_value_layer = self.value(hidden_states)#wm,768->768

        query_layer = self.transpose_for_scores(mixed_query_layer)#wm，(bs,12,197,64)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))#将q向量和k向量进行相乘（bs,12,197,197)
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)#将结果除以向量维数的开方
        attention_probs = self.softmax(attention_scores)#将得到的分数进行softmax,得到概率
        weights = attention_probs if self.vis else None#wm,实际上就是权重
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)#将概率与内容向量相乘
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)#wm,(bs,197)+(768,)=(bs,197,768)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output, weights#wm,(bs,197,768),(bs,197,197)

#3.构建前向传播神经网络
#两个全连接神经网络，中间加了激活函数
class Mlp(nn.Module):
    def __init__(self, config):
        super(Mlp, self).__init__()
        self.fc1 = nn.Linear(config.hidden_size, config.mlp_dim)#wm,786->3072
        self.fc2 = nn.Linear(config.mlp_dim, config.hidden_size)#wm,3072->786
        self.act_fn = nn.functional.gelu#wm,激活函数
        self.dropout = nn.Dropout(config.dropout_rate)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)#wm,786->3072
        x = self.act_fn(x)#激活函数
        x = self.dropout(x)#wm,丢弃
        x = self.fc2(x)#wm3072->786
        x = self.dropout(x)
        return x

#4.构建编码器的可重复利用的Block()模块：每一个block包含了self-attention模块和MLP模块
class Block(nn.Module):
    def __init__(self, config, vis):
        super(Block, self).__init__()
        self.hidden_size = config.hidden_size#wm,768
        self.attention_norm = nn.LayerNorm(config.hidden_size, eps=1e-6)#wm，层归一化
        self.ffn_norm = nn.LayerNorm(config.hidden_size, eps=1e-6)
        
        self.ffn = Mlp(config)
        self.attn = Attention(config, vis)

    def forward(self, x):
        h = x
        x = self.attention_norm(x)
        x, weights = self.attn(x)
        x = x + h#残差结构

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h#残差结构
        return x, weights

import copy
#5.构建Encoder模块，该模块实际上就是堆叠N个Block模块
class Encoder(nn.Module):
    def __init__(self, vis):
        super(Encoder, self).__init__()
        self.vis = vis
        self.layer = nn.ModuleList()
        self.encoder_norm = nn.LayerNorm(config.hidden_size, eps=1e-6)
        for _ in range(config.num_layers):
            layer = Block(config, vis)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, hidden_states):
        attn_weights = []
        for layer_block in self.layer:
            hidden_states, weights = layer_block(hidden_states)
            if self.vis:
                attn_weights.append(weights)
        encoded = self.encoder_norm(hidden_states)
        return encoded, attn_weights

#6构建transformers完整结构，首先图片被embedding模块编码成序列数据，然后送入Encoder中进行编码
class Transformer(nn.Module):
    def __init__(self, vis):
        super(Transformer, self).__init__()
        self.embeddings = Embeddings(config.image_size[0])#wm,对一幅图片进行切块编码，得到的是（bs,n_patch+1（196）,每一块的维度（768））
        self.encoder = Encoder(vis)

    def forward(self, input_ids):
        embedding_output = self.embeddings(input_ids)#wm,输出的是（bs,196,768)
        encoded, attn_weights = self.encoder(embedding_output)#wm,输入的是（bs,196,768)
        return encoded, attn_weights#输出的是（bs,197,768）

#7构建VisionTransformer，用于图像分类
class VisionTransformer(nn.Module):
    def __init__(self, num_classes=2, zero_head=False, vis=False):
        super(VisionTransformer, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.classifier = config.classifier

        self.transformer = Transformer(vis)
        self.head = nn.Linear(config.hidden_size, num_classes)#wm,768-->2

    def forward(self, x, labels=None):
        x, attn_weights = self.transformer(x)
        logits = self.head(x[:, 0])
        return logits

        # #如果传入真实标签，就直接计算损失值
        # if labels is not None:
        #     loss_fct = CrossEntropyLoss()
        #     loss = loss_fct(logits.view(-1, self.num_classes), labels.view(-1))
        #     return loss
        # else:
        #     return logits, attn_weights

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

def adjust_learning_rate(optimizer, epoch):
    lr = config.lr * (0.1 ** (epoch // 10))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    print('learning rate update to %f', lr)

def main():
    # load data from npy
    train_list = np.load('./npy/train_list.npy')
    val_list = np.load('./npy/val_list.npy')

    train_data = dataset(train_list, transform=init_data.train_transforms)
    val_data = dataset(val_list, transform=init_data.val_transforms)

    print(len(val_list))

    train_loader = torch.utils.data.DataLoader(dataset = train_data, batch_size=config.batch_size, shuffle=True )
    val_loader = torch.utils.data.DataLoader(dataset = val_data, batch_size=config.batch_size, shuffle=True)

    model = VisionTransformer().to(device)
    model.train()

    optimizer = optim.Adam(params = model.parameters(),lr=config.lr)
    # criterion = nn.BCEWithLogitsLoss()
    criterion = nn.CrossEntropyLoss()

    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    for epoch in range(config.epochs):
        epoch_loss = 0
        epoch_accuracy = 0
        
        for data, label in tqdm(train_loader):
            data = data.to(device)
            label = label.to(device)
            
            output = model(data)
            # output = output.squeeze(dim=-1)
            # loss = criterion(output, label.to(torch.float32))

            # transformer loss:
            # output = output.view(-1, 2)
            # label = output.view(-1)
            loss = criterion(output, label)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            acc = ((output.argmax(dim=1) == label).float().mean())
            # acc = ((output.round() == label).float().mean())
            epoch_accuracy += acc/len(train_loader)
            epoch_loss += loss/len(train_loader)
            
        print('Epoch : {}, train accuracy : {}, train loss : {}'.format(epoch+1, epoch_accuracy,epoch_loss))
        train_losses.append(epoch_loss.item())
        train_accuracies.append(epoch_accuracy.item())
        
        with torch.no_grad():
            epoch_val_accuracy=0
            epoch_val_loss=0
            for data, label in val_loader:
                data = data.to(device)
                label = label.to(device)
                
                val_output = model(data)
                # val_output = val_output.squeeze(dim=-1)
                # val_loss = criterion(val_output, label.to(torch.float32))
                val_loss = criterion(val_output, label)
                
                acc = ((val_output.argmax(dim=1) == label).float().mean())
                # acc = ((val_output.round() == label).float().mean())
                epoch_val_accuracy += acc/ len(val_loader)
                epoch_val_loss += val_loss/ len(val_loader)
                
            print('Epoch : {}, val_accuracy : {}, val_loss : {}'.format(epoch+1, epoch_val_accuracy,epoch_val_loss))
            val_losses.append(epoch_val_loss.item())
            val_accuracies.append(epoch_val_accuracy.item())

    plot_training(config.loss_img_save, config.acc_img_save, config.epochs, train_losses, val_losses, train_accuracies, val_accuracies)

    torch.save(model.state_dict(), config.model_name)

    # total_samples = 0
    # correct_predictions = 0
    # total_loss = 0

    # model.eval()
    # with torch.no_grad():
    #     for data, labels in val_data:
    #         data = data.to(device)
    #         labels = labels.to(device)

    #         preds = model(data)
    #         _, predicted = torch.max(preds.data, 1)

    #         total_samples += labels.size(0)
    #         correct_predictions += (predicted == labels).sum().item()

    #         loss = criterion(preds, labels)
    #         total_loss += loss.item()

    # accuracy = correct_predictions / total_samples
    # average_loss = total_loss / len(val_data)
    # print(f' === Test Accuracy: {accuracy:.4f}')
    # print(f' === Test Loss: {average_loss:.4f}')

if __name__ == "__main__":
    main()