import os
from typing import Dict, Tuple

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from ewc.constants import *


def load_datasets(
    data_dir: str = DEFAULT_DATA_DIR,
) -> Tuple[Dict[str, DataLoader], Dict[str, DataLoader]]:
    """
    Load MNIST and Fashion-MNIST datasets and create dataloaders.

    Args:
        data_dir: Directory to store the datasets

    Returns:
        Tuple of (train_loaders, test_loaders) dictionaries with DataLoaders
    """
    try:
        # Define transformations for both datasets
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((MNIST_MEAN,), (MNIST_STD,))]
        )

        # Create dataset directory if it doesn't exist
        os.makedirs(data_dir, exist_ok=True)

        # Load MNIST dataset
        mnist_train = datasets.MNIST(
            data_dir, train=True, download=True, transform=transform
        )
        mnist_test = datasets.MNIST(data_dir, train=False, transform=transform)

        # Load Fashion-MNIST dataset
        fashion_train = datasets.FashionMNIST(
            data_dir, train=True, download=True, transform=transform
        )
        fashion_test = datasets.FashionMNIST(data_dir, train=False, transform=transform)

        # Create data loaders
        train_loaders = {
            "mnist": DataLoader(mnist_train, batch_size=BATCH_SIZE, shuffle=True),
            "fashion": DataLoader(fashion_train, batch_size=BATCH_SIZE, shuffle=True),
        }

        test_loaders = {
            "mnist": DataLoader(mnist_test, batch_size=TEST_BATCH_SIZE),
            "fashion": DataLoader(fashion_test, batch_size=TEST_BATCH_SIZE),
        }

        print("Datasets loaded successfully")
        return train_loaders, test_loaders

    except Exception as e:
        print(f"Error loading datasets: {e}")
        raise
