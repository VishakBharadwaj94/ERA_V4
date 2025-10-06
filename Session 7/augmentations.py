"""
augmentations.py - Albumentations transforms for CIFAR-10
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np


# CIFAR-10 dataset statistics
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)


class AlbumentationsTransform:
    """Wrapper to make Albumentations compatible with PyTorch datasets"""
    def __init__(self, transform):
        self.transform = transform
    
    def __call__(self, img):
        img = np.array(img)
        augmented = self.transform(image=img)
        return augmented['image']


def get_train_transforms():
    """
    Training augmentations using Albumentations:
    - HorizontalFlip
    - ShiftScaleRotate
    - CoarseDropout (16x16 hole filled with dataset mean)
    """
    return AlbumentationsTransform(A.Compose([
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
        A.CoarseDropout(
            num_holes_range=(1, 1),
            hole_height_range=(16, 16), 
            hole_width_range=(16, 16),
            fill=tuple([int(x * 255.0) for x in CIFAR_MEAN]),
            p=0.5
        ),
        A.Normalize(mean=CIFAR_MEAN, std=CIFAR_STD),
        ToTensorV2()
    ]))


def get_test_transforms():
    """Test transforms (normalization only)"""
    return AlbumentationsTransform(A.Compose([
        A.Normalize(mean=CIFAR_MEAN, std=CIFAR_STD),
        ToTensorV2()
    ]))