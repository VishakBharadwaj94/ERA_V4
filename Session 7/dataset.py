"""
dataset.py - CIFAR-10 dataset loading and preparation
"""

import torch
from torchvision import datasets
from torch.utils.data import DataLoader
from augmentations import get_train_transforms, get_test_transforms


def get_cifar10_loaders(batch_size=128, num_workers=4):
    """
    Create CIFAR-10 train and test data loaders with augmentation
    
    Args:
        batch_size: Batch size for training and testing
        num_workers: Number of worker processes for data loading
        
    Returns:
        train_loader, test_loader
    """
    
    # Set seed for reproducibility
    SEED = 1
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
    
    # Training dataset with augmentation
    train_dataset = datasets.CIFAR10(
        root='./data', 
        train=True, 
        download=True,
        transform=get_train_transforms()
    )
    
    # Test dataset without augmentation
    test_dataset = datasets.CIFAR10(
        root='./data', 
        train=False, 
        download=True,
        transform=get_test_transforms()
    )
    
    # Determine dataloader arguments based on CUDA availability
    cuda = torch.cuda.is_available()
    dataloader_args = dict(
        shuffle=True, 
        batch_size=batch_size, 
        num_workers=num_workers, 
        pin_memory=True
    ) if cuda else dict(
        shuffle=True, 
        batch_size=64
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, **dataloader_args)
    test_loader = DataLoader(test_dataset, **dataloader_args)
    
    return train_loader, test_loader