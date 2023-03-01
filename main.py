from typing import Tuple, List

import wandb
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
from transformers import ViTImageProcessor

from model import ClassificationModel

run = wandb.init("vit-cifar10")

batch_size = 30
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
model.to(device)

# Set optimizer
optim = torch.optim.Adam(model.parameters(), lr=1e-4)

# Set Loss function
criterion = torch.nn.CrossEntropyLoss()

# Train
for epoch in range(1000):
    for inputs, labels in trainloader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        # Forward
        output = model(inputs)  # [batch_size, num_classes] in CIFAR-10
        loss = criterion(output, labels)  # Calculate loss

        # Backward
        optim.zero_grad()  # Initialize gradient to zero
        loss.backward()  # Calculate gradient
        optim.step()  # Update parameters

        # print accuracy
        y_hat = torch.argmax(output, dim=-1)
        correct = torch.sum(y_hat == labels)
        print(f'epoch Accuracy: {correct / len(labels)}')
        wandb.log({"training Accuracy": correct / len(labels)})

        # Print loss
        print(f'epoch Loss: {loss.item()}')
        wandb.log({"training Loss": loss.item()})

        # # Save model
        # torch.save(model.state_dict(), 'model.pth')

    # Validate
    with torch.no_grad():
        correct = 0
        total = 0
        for inputs, labels in testloader:
            output = model(inputs)
            y_hat = torch.argmax(output, dim=-1)
            total += len(labels)
            correct += torch.sum(y_hat == labels)

        # Print accuracy
        print(f'Epoch : {epoch} Accuracy: {correct / total}')
        wandb.log({"validation Accuracy": correct / total})