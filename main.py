import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
from transformers import ViTImageProcessor

from model import ClassificationModel

transform = transforms.Compose([transforms.ToTensor()])

# Load Dataset
trainset = CIFAR10(root='./data', train=True,
                   download=True, transform=transform)
testset = CIFAR10(root='./data', download=True,
                  train=False, transform=transform)

processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')
def collate_fn(batch: List[Tuple[torch.Tensor, int]]):
    # Get batch
    inputs = [item[0] for item in batch]
    inputs = processor(inputs, return_tensors="pt")
    labels = torch.Tensor([item[1] for item in batch]).long()

    return inputs, labels


# Make DataLoader
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
testloader = DataLoader(testset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

# Setmodel
model = ClassificationModel(num_classes=10)

# Set optimizer
optim = torch.optim.Adam(model.parameters(), lr=1e-4)
