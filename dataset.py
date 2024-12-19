import torch                                    # Core PyTorch library for tensor computations.
from torchvision import datasets                # Provides datasets including FashionMNIST.
from torchvision.transforms import ToTensor     # Converts image data to PyTorch tensors.
from torch.utils.data import DataLoader         # Simplifies batch loading of datasets.

class FashionMNISTDataset:
    """
    Handles loading the FashionMNIST dataset using PyTorch's utilities.
    This class encapsulates the functionality of downloading, transforming,
    and batching the dataset for training and testing.

    Args:
        batch_size (int): The number of samples to load per batch. Default is 128.
        num_workers (int): The number of subprocesses to use for data loading. 
                           Default is 1 (suitable for small datasets or systems with limited CPUs).
    """
    def __init__(self, batch_size=128, num_workers=1):
        # Initialize parameters for data loading
        self.batch_size = batch_size    # Number of samples per batch
        self.num_workers = num_workers  # Number of subprocesses for parallel data loading

    def load_data(self):
        """
        Downloads and prepares the FashionMNIST dataset. Applies transformations to convert
        image data into PyTorch tensors and provides DataLoaders for both training and testing.

        The dataset contains grayscale images of size 28x28, grouped into 10 classes
        (e.g., T-shirts, trousers, sneakers). Each class has a balanced number of samples.

        Returns:
            dict: A dictionary containing DataLoaders for 'train' and 'test' datasets.
                  - 'train': DataLoader for the training set with shuffling enabled.
                  - 'test': DataLoader for the testing set with no shuffling.
        """
        # Load the training dataset with transformations applied
        train_data = datasets.FashionMNIST(
            root='data',                # Directory to store/download the dataset
            train=True,                 # Flag to indicate this is the training set
            transform=ToTensor(),       # Transform the image data into PyTorch tensors
            download=True,              # Download the dataset if it's not already available
        )

        # Load the testing dataset with similar transformations
        test_data = datasets.FashionMNIST(
            root='data',                # Same root directory as the training set
            train=False,                # Flag to indicate this is the testing set
            transform=ToTensor(),       # Apply the same tensor transformation
        )

        # Create DataLoaders for both the training and testing datasets
        loaders = {
            'train': DataLoader(
                train_data,                     # Pass the training dataset
                batch_size=self.batch_size,     # Use the specified batch size
                shuffle=True,                   # Shuffle the training data for better learning
                num_workers=self.num_workers,   # Number of parallel data loading workers
            ),
            'test': DataLoader(
                test_data,                      # Pass the testing dataset
                batch_size=self.batch_size,     # Use the same batch size
                shuffle=False,                  # No need to shuffle the test data
                num_workers=self.num_workers,   # Number of parallel data loading workers
            ),
        }

        return loaders  # Return the DataLoaders as a dictionary
