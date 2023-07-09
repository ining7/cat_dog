from torchvision import transforms

import numpy as np
import os
import shutil
import glob

from sklearn.model_selection import train_test_split

import config

#data Augumentation
train_transforms = transforms.Compose([
        transforms.Resize(config.image_size),
        transforms.RandomResizedCrop(config.image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

val_transforms = transforms.Compose([
        transforms.Resize(config.image_size),
        transforms.RandomResizedCrop(config.image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])


test_transforms = transforms.Compose([   
    transforms.Resize(config.image_size),
    transforms.RandomResizedCrop(config.image_size),
    transforms.RandomHorizontalFlip(),
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

def move_image_parent_folder(folder_path):
    if not os.path.isdir(folder_path):
        print(f"folder_path '{folder_path}' not found.")
        return
    subfolders = [subfolder for subfolder in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, subfolder))]
    for subfolder in subfolders:
        subfolder_path = os.path.join(folder_path, subfolder)
        files = [file for file in os.listdir(subfolder_path) if os.path.isfile(os.path.join(subfolder_path, file))]
        for file in files:
            file_path = os.path.join(subfolder_path, file)
            shutil.move(file_path, folder_path)
        shutil.rmtree(subfolder_path)

def main():
    train_dir_source = '../input/archive/train_after_crop'
    val_dir_source = '../input/archive/validation_with_group'
    test_dir = '../data/test'
    train_dir = '../data/archive_train'
    val_dir = '../data/archive_val'
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    # move_image_parent_folder(train_dir_source)
    # move_image_parent_folder(val_dir_source)

    rename_sort_files(train_dir_source, train_dir)
    rename_sort_files(val_dir_source, val_dir)
    print("Renamed and reordered the dataset.")

    train_list = glob.glob(os.path.join(train_dir,'*.jpg'))
    val_list = glob.glob(os.path.join(val_dir,'*.jpg'))
    test_list = glob.glob(os.path.join(test_dir, '*.jpg'))

    # save data to npy
    os.makedirs('./npy', exist_ok=True)
    np.save('./npy/train_list.npy', train_list)
    np.save('./npy/val_list.npy', val_list)
    np.save('./npy/test_list.npy', test_list)

if __name__ == "__main__":
    main()