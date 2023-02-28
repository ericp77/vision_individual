import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms

batch_size = 4

transform = transforms.Compose([transforms.ToTensor()])

# Load Dataset
trainset = CIFAR10(root='./data', train=True,
                   download=True, transform=transform)
testset = CIFAR10(root='./data', download=True,
                  train=False, transform=transform)

# Make DataLoader
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
testloader = DataLoader(testset, batch_size=batch_size, shuffle=True)

