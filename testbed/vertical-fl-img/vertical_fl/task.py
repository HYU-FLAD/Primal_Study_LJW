from logging import WARN
import numpy as np
import torch
import torch.nn as nn
from flwr.common.logger import log

from torchvision.datasets import CIFAR10, MNIST
from torchvision import transforms

from flwr_datasets.partitioner import IidPartitioner
from datasets import Dataset

NUM_VERTICAL_SPLITS = 3
CLIENT_EMBEDDING_SIZE = 16 # Size of the feature embedding from each client

# ADDED: Define two types of client models
class ClientModelMLP(nn.Module):
    """A simple MLP model for a slice of flattened image data."""
    def __init__(self, input_size, output_size=CLIENT_EMBEDDING_SIZE):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, output_size)

    def forward(self, x):
        # x는 GPU에 있는 미니 배치 데이터
        if x.shape[1] == 1024: # CIFAR-10 slice
            x = x.view(-1, 1, 32, 32)
        elif x.shape[1] == 261: # MNIST slice (783/3)
            pad = torch.zeros(x.shape[0], 28, device=x.device)
            
            x = torch.cat([x, pad], dim=1)
            x = x.view(-1, 1, 17, 17)
        else:
            raise ValueError("Unsupported input size for CNN model")

        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5) # Flatten
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)

class ClientModelCNN(nn.Module):
    """A simple CNN model for a slice of image data."""
    def __init__(self, input_channels, output_size=CLIENT_EMBEDDING_SIZE):
        super().__init__()
        # We assume the input slice can be reshaped into a square-like image
        # For CIFAR-10 (3072 pixels / 3 = 1024), this can be reshaped to 32x32.
        # For MNIST (784 pixels), this is not perfectly divisible by 3.
        # We will handle padding/truncating in load_data.
        self.conv1 = nn.Conv2d(input_channels, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # This will be dynamically calculated
        self.fc1 = nn.Linear(16 * 5 * 5, 120) # Example for 32x32 input
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        # Assuming x is a batch of flattened slices
        # The reshape logic depends on the original image size and split
        if x.shape[1] == 1024: # CIFAR-10 slice
            x = x.view(-1, 1, 32, 32)
        elif x.shape[1] == 261: # MNIST slice (783/3)
             # Pad to make it 17x17 = 289
            pad = torch.zeros(x.shape[0], 28)
            x = torch.cat([x, pad], dim=1)
            x = x.view(-1, 1, 17, 17)
        else:
            raise ValueError("Unsupported input size for CNN model")

        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5) # Flatten
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)

def _load_raw_dataset(name: str):
    """Load raw CIFAR-10 or MNIST dataset."""
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    if name == "CIFAR10":
        trainset = CIFAR10(root="./data", train=True, download=True, transform=transform)
        # For CIFAR-10, data is (N, 3, 32, 32). We'll treat the 3 channels as one dimension for flattening.
        data = trainset.data.reshape(len(trainset), -1)
    elif name == "MNIST":
        trainset = MNIST(root="./data", train=True, download=True, transform=transform)
        data = trainset.data.numpy().reshape(len(trainset), -1)
    else:
        raise ValueError(f"Dataset {name} not supported.")
    
    # Normalize pixel values to [0, 1]
    data = data / 255.0
    return data, np.array(trainset.targets)

def load_server_labels(dataset_name: str):
    """Load only labels for the server."""
    _, labels = _load_raw_dataset(dataset_name)
    return labels

def load_data(partition_id: int, num_partitions: int, dataset_name: str):
    """Partition the image data vertically and then horizontally."""
    if num_partitions != NUM_VERTICAL_SPLITS:
        log(WARN, "This example is designed for num_partitions to be exactly NUM_VERTICAL_SPLITS.")

    # Load whole dataset
    data, labels = _load_raw_dataset(dataset_name)

    # For MNIST (784 features), pad to be divisible by 3 (786)
    if dataset_name == "MNIST" and data.shape[1] % NUM_VERTICAL_SPLITS != 0:
        padding = np.zeros((data.shape[0], NUM_VERTICAL_SPLITS - (data.shape[1] % NUM_VERTICAL_SPLITS)))
        data = np.concatenate([data, padding], axis=1)

    # Vertical Split
    v_partitions = np.array_split(data, NUM_VERTICAL_SPLITS, axis=1)
    v_split_id = np.mod(partition_id, NUM_VERTICAL_SPLITS)
    v_partition_data = v_partitions[v_split_id]

    # Combine with labels for horizontal partitioning
    full_v_partition = {"features": list(v_partition_data), "label": list(labels)}
    dataset = Dataset.from_dict(full_v_partition)

    # Horizontal Split (only one horizontal partition per vertical split in this setup)
    num_h_partitions = int(np.ceil(num_partitions / NUM_VERTICAL_SPLITS))
    if num_h_partitions > 1:
        log(WARN, "Horizontal partitioning is not fully utilized in this simplified VFL setup.")

    # In this VFL setup, each client in a vertical group gets all samples.
    # So we don't need a horizontal partitioner. We return the full vertical slice.
    # Note: In a more complex scenario, you would partition this `dataset` horizontally.
    h_partition_index = partition_id // NUM_VERTICAL_SPLITS
    # Since num_partitions == NUM_VERTICAL_SPLITS, h_partition_index is always 0.
    
    return v_partition_data, v_split_id