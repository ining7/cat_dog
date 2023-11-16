import torch
import os

image_size = (250, 250)
channel_count = 3

lr = 0.0001  # learning_rate
batch_size = 32  # we will use mini-batch method
epochs = 500  # How much to train a model

params_str = "{}_{}_{}".format(lr, batch_size, epochs)

img_path = f'../image/{params_str}'
os.makedirs(img_path, exist_ok=True)
loss_img_save = os.path.join(img_path, 'loss.png')
acc_img_save = os.path.join(img_path, 'acc.png')

model_name = f'../check_point/{params_str}.pth'

# check device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device == 'cuda':
    print("the device is gpu")
else:
    print("the device is cpu")
