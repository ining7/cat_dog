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

def main():
    base_dir = '../input/dogs-vs-cats-redux-kernels-edition'
    train_zip_path = os.path.join(base_dir, 'train.zip')
    base_temp_dir = '../temp_data/base_temp_dir'
    train_dir = '../data/train'
    test_dir = '../data/test'
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    with zipfile.ZipFile(train_zip_path) as train_zip:
        train_zip.extractall(base_temp_dir)
    print("Extracted files to {}.".format(base_temp_dir))

    train_temp_dir = '../temp_data/train_temp_dir'
    test_temp_dir = '../temp_data/test_temp_dir'
    os.makedirs(train_temp_dir, exist_ok=True)
    os.makedirs(test_temp_dir, exist_ok=True)

    flag = 1
    for root, dirs, files in os.walk(base_temp_dir):
        for file in files:
            temp_path = os.path.join(base_temp_dir,'train', file)
            train_path = os.path.join(train_temp_dir, file)
            test_path = os.path.join(test_temp_dir, file)
            if flag:
                shutil.copyfile(temp_path, test_path)
            else:
                shutil.copyfile(temp_path, train_path)
            flag ^= 1
    print("Split the data into training and testing sets.")

    rename_sort_files(train_temp_dir, train_dir)
    rename_sort_files(test_temp_dir, test_dir)
    print("Renamed and reordered the dataset.")

    shutil.rmtree(os.path.dirname(base_temp_dir))
    print("Deleted temporary files.")

    train_list = glob.glob(os.path.join(train_dir,'*.jpg'))
    test_list = glob.glob(os.path.join(test_dir, '*.jpg'))

    train_list, val_list = train_test_split(train_list, test_size=0.2)

    # save data to npy
    os.makedirs('./npy', exist_ok=True)
    np.save('./npy/train_list.npy', train_list)
    np.save('./npy/val_list.npy', val_list)
    np.save('./npy/test_list.npy', test_list)

if __name__ == "__main__":
    main()