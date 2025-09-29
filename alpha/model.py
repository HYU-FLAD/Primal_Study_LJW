from torch import nn
import torch.nn.functional as F
from torchvision import models

class SimpleCNN(nn.Module):
    """A simple CNN for CIFAR-10."""
    def __init__(self, args):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(args.num_channels, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, args.num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def get_model(args):
    """
    Returns the appropriate model based on the model name in args.
    The model architecture is dynamically adjusted based on the dataset.
    """
    model_name = args.model.lower()
    
    if model_name in ['resnet18', 'resnet50']:
        if model_name == 'resnet18':
            model = models.resnet18(weights=None)
        else:
            model = models.resnet50(weights=None)

        # Dynamically adapt the model for small image datasets like CIFAR
        if args.dataset in ['cifar10', 'cifar100']:
            print(f"Adapting {model_name.upper()} for {args.dataset.upper()}...")
            # Modify the first convolutional layer to preserve spatial information
            model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            # Remove the initial max pooling layer
            model.maxpool = nn.Identity()
        else:
            print(f"Using standard {model_name.upper()} for large images...")

        # Adjust the final fully-connected layer to the number of classes
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, args.num_classes)

    elif model_name == 'vit':
        print("Loading Vision Transformer (ViT-B/16)...")
        model = models.vit_b_16(weights='IMAGENET1K_V1' if args.pretrained else None)
        
        # Replace the final classification head
        num_ftrs = model.heads.head.in_features
        model.heads.head = nn.Linear(num_ftrs, args.num_classes)

    elif model_name == "simple_cnn":
        model = SimpleCNN(args)

    else:
        # You can add other models like the SimpleCNN here if needed
        raise NotImplementedError(f"Model '{args.model}' is not supported.")
    
    # Attach properties for framework compatibility
    model.name = 'server'
    model.len = 0
    
    return model
