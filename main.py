import torch

from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms

transform = transforms.Compose([transforms.ToTensor()])

# Load Dataset
trainset = CIFAR10(root='./data', train=True,
                   download=True, transform=transform)
testset = CIFAR10(root='./data', download=True,
                  train=False, transform=transform)