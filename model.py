from torch import nn
from transformers import ViTModel


class ClassificationModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        # Get pretrained model from HuggingFace
        self.model = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')

        # Add Classifier
        self.classifier = nn.Linear(self.model.config.hidden_size, num_classes)  # [768, 10] in CIFAR-10
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, input_image):
        # Get features from pretrained model
        features = self.model(**input_image).last_hidden_state[:, 0, :]

        # Get output
        logits = self.classifier(features)
        output = self.softmax(logits)

        return output
