import os
import torch
import torchvision.transforms as transforms

from train_utils.myDataset import MyDataSet
from train_utils.utilsTrain import read_split_data


def get_train_transform():
    return transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def get_val_transform():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def create_data_loaders_split(args, val_ratio=0.1):
    # 读取数据并划分训练集和验证集
    full_train_images_path, full_train_images_label, _, _ = read_split_data(args.data_path)

    # Define data transformations
    train_transform = get_train_transform()
    val_transform = get_val_transform()

    # 实例化训练数据集
    full_train_dataset = MyDataSet(images_path=full_train_images_path,
                                   images_class=full_train_images_label)

    # Calculate the number of samples in the validation set
    num_val_samples = int(val_ratio * len(full_train_dataset))
    num_train_samples = len(full_train_dataset) - num_val_samples

    # Split the training dataset into train and validation sets
    train_dataset, val_dataset = torch.utils.data.random_split(full_train_dataset, [num_train_samples, num_val_samples])

    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))

    # Apply the corresponding transformations to train and validation sets
    train_dataset.dataset.transform = train_transform
    val_dataset.dataset.transform = val_transform

    # 创建训练集和验证集的DataLoader
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=nw)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             num_workers=nw)

    return train_loader, val_loader


def create_data_loaders_train_val(args):
    # 读取数据并划分训练集和验证集
    train_images_path, train_images_label, _, _ = read_split_data(args.data_path)
    val_images_path, val_images_label, _, _ = read_split_data(args.data_val_path)

    # Define data transformations
    train_transform = get_train_transform()
    val_transform = get_val_transform()

    # 实例化训练数据集
    train_dataset = MyDataSet(images_path=train_images_path,
                              images_class=train_images_label,
                              transform=train_transform)
    # 实例化val数据集
    val_dataset = MyDataSet(images_path=val_images_path,
                            images_class=val_images_label,
                            transform=val_transform)

    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))


    # 创建训练集和验证集的DataLoader
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=nw)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             num_workers=nw)

    return train_loader, val_loader
