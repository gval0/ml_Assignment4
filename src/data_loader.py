import torch
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

class FERDataset(Dataset):
    def __init__(self, csv_file, train=True, transform=None):
        self.data = pd.read_csv(csv_file)
        self.train = train
        self.transform = transform
        self.emotion_labels = {
            0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy',
            4: 'Sad', 5: 'Surprise', 6: 'Neutral'
        }
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if self.train:
            pixels = self.data.iloc[idx]['pixels']
            emotion = self.data.iloc[idx]['emotion']
        else:
            pixels = self.data.iloc[idx]['pixels']
            emotion = -1
        image = np.array([int(pixel) for pixel in pixels.split()], dtype=np.uint8)
        image = image.reshape(48, 48)
        image = Image.fromarray(image)
        
        if self.transform:
            image = self.transform(image)
        if self.train:
            return image, emotion
        else:
            return image, idx

def get_data_transforms(augment=True):
    if augment:
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
    else:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
    
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    return train_transform, val_transform

def create_data_loaders(csv_file, batch_size=32, val_split=0.2, 
                       train_transform=None, val_transform=None, num_workers=2):
    if train_transform is None or val_transform is None:
        train_transform, val_transform = get_data_transforms(augment=True)
    full_dataset = FERDataset(csv_file, train=True, transform=train_transform)
    dataset_size = len(full_dataset)
    val_size = int(val_split * dataset_size)
    train_size = dataset_size - val_size
    
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    val_dataset.dataset.transform = val_transform
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    return train_loader, val_loader

def create_test_loader(csv_file, batch_size=32, num_workers=2):
    _, test_transform = get_data_transforms(augment=False)
    test_dataset = FERDataset(csv_file, train=False, transform=test_transform)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    return test_loader