"""
Image preprocessing functions
"""

import torch
import numpy as np
import os
from torchvision import transforms, datasets

RESNET_NORM_MEAN = [0.485, 0.456, 0.406]
RESNET_NORM_STD = [0.229, 0.224, 0.225]


def denormalize(img_tensor: torch.Tensor):
    """
    Denormalizing step for visualization
    img_tensor: (C,H,W) tensor
    returns: (H,W,C) tensor
    """
    img = img_tensor.permute(1, 2, 0).cpu().numpy()
    img = img * RESNET_NORM_STD + RESNET_NORM_MEAN
    img = np.clip(img, 0, 1)
    return img


def get_data_loaders(batch_size=10, data_dir="data"):
    """
    Using similar preprocessing as
    https://www.kaggle.com/code/carloalbertobarbano/vgg16-transfer-learning-pytorch
    """

    transformations = {
        "train": transforms.Compose(
            [
                transforms.Grayscale(num_output_channels=3),
                transforms.RandomHorizontalFlip(),
                transforms.RandomResizedCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(RESNET_NORM_MEAN, RESNET_NORM_STD),
            ]
        ),
        "test": transforms.Compose(
            [
                transforms.Grayscale(num_output_channels=3),
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(RESNET_NORM_MEAN, RESNET_NORM_STD),
            ]
        ),
        "val": transforms.Compose(
            [
                transforms.Grayscale(num_output_channels=3),
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(RESNET_NORM_MEAN, RESNET_NORM_STD),
            ]
        ),
    }

    img_datasets = {
        x: datasets.ImageFolder(os.path.join(data_dir, x), transform=transformations[x])
        for x in transformations.keys()
    }

    dataloaders = {
        x: torch.utils.data.DataLoader(
            img_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4
        )
        for x in transformations.keys()
    }

    return dataloaders
