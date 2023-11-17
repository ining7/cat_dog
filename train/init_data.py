from torchvision import transforms

import numpy as np
import os
import zipfile
import shutil
import glob

from sklearn.model_selection import train_test_split

import config

#data Augumentation
train_transforms = transforms.Compose([
        transforms.Resize(config.image_size),
        # transforms.RandomResizedCrop(config.image_size),
        # transforms.RandomHorizontalFlip(),
        # transforms.RandomRotation(15),
        # transforms.ColorJitter(brightness=1, contrast=1, hue=0.5, saturation=0.5),
        transforms.ToTensor(),
    ])

val_transforms = transforms.Compose([
        transforms.Resize(config.image_size),
        # transforms.RandomResizedCrop(config.image_size),
        # transforms.RandomHorizontalFlip(),
        # transforms.RandomRotation(15),
        # transforms.ColorJitter(brightness=1, contrast=1, hue=0.5, saturation=0.5),
        transforms.ToTensor(),
    ])


test_transforms = transforms.Compose([   
    transforms.Resize(config.image_size),
    # transforms.RandomResizedCrop(config.image_size),
    # transforms.RandomHorizontalFlip(),
    # transforms.RandomRotation(15),
    # transforms.ColorJitter(brightness=1, contrast=1, hue=0.5, saturation=0.5),
    transforms.ToTensor()
    ])

def rename_sort_files(src_dir, dst_dir):
    for root, dirs, files in os.walk(src_dir):
        for i, file in enumerate(files):
            old_file_path = os.path.join(src_dir, file)
            name_part = file.split('.')[0]
            new_file_name = f'{name_part}.{i}.jpg'
            new_file_path = os.path.join(dst_dir, new_file_name)
            shutil.copyfile(old_file_path, new_file_path)

def main():
    base_dir = '../../archive'
    train_dir = '../data/25000images/train'
    val_dir = '../data/25000images/val'
    test_dir = '../data/test'
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
 
    train_temp_dir = '../temp_data/train_temp_dir'
    test_temp_dir = '../temp_data/test_temp_dir'
    os.makedirs(train_temp_dir, exist_ok=True)
    os.makedirs(test_temp_dir, exist_ok=True)

    train_list = glob.glob(os.path.join(train_dir,'*.jpg'))
    val_list = glob.glob(os.path.join(val_dir,'*.jpg'))

    print(len(train_list), len(val_list))

    # save data to npy
    os.makedirs('./npy', exist_ok=True)
    np.save('./npy/train_list.npy', train_list)
    np.save('./npy/val_list.npy', val_list)
    np.save('./npy/test_list.npy', val_list)

if __name__ == "__main__":
    main()