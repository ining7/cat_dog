import torch
import os

lr = 0.002  # learning_rate
batch_size = 200  # we will use mini-batch method
epochs = 20  # How much to train a model

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
