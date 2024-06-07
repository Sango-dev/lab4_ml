import yaml
import os
import requests
import tarfile
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import random_split
import torch
import torch.nn as nn
import torch.optim as optim

with open("params.yaml", "r") as f:
    config = yaml.safe_load(f)
 


image_size = (224, 224)

def load_and_split_data(data_dir):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(image_size),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Load the train dataset
    train_dataset =datasets.OxfordIIITPet(root=data_dir, split="trainval", download=True, transform=transform)

    # Load the test dataset
    test_dataset =datasets.OxfordIIITPet(root=data_dir, split = "test", download=True, transform=transform)

    # Calculate the validation split size
    total_size = len(train_dataset)
    val_size = int(config["dataset"]["val_split"] * total_size)
    train_size = total_size - val_size

    # Split the train dataset into train and validation sets
    train_dataset, val_dataset = random_split(
        train_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(config["dataset"]["random_seed"])
    )
    return train_dataset, val_dataset, test_dataset




if __name__ == "__main__":
    destination_folder = config['data']['local_dir']
    train_dataset, val_dataset, test_dataset = load_and_split_data(destination_folder)

    # Save the split datasets to pickle files
    import pickle
    with open('data/train_dataset.pkl', 'wb') as f:
        pickle.dump(train_dataset, f)
    with open('data/val_dataset.pkl', 'wb') as f:
        pickle.dump(val_dataset, f)
    with open('data/test_dataset.pkl', 'wb') as f:
        pickle.dump(test_dataset, f)

 
