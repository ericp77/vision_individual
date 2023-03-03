from torch import nn
from transformers import ViTModel
from transformers import ViTConfig

"""
https://huggingface.co/google/vit-base-patch16-224-in21k
https://pytorch.org/tutorials/beginner/nn_tutorial.html
https://github.com/features/copilot
"""

# Without Pretrained
class ClassificationModel(nn.Module):

    """
    # With Pretraining
    def __init__(self, num_classes):                                                                              │|-------------------------------+----------------------+----------------------+
        super().__init__()                                                                                        │| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
                                                                                                                  │| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
        # Get pretrained model from HuggingFace                                                                   │|                               |                      |               MIG M. |
        self.model = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')                                │|===============================+======================+======================|
                                                                                                                  │|   0  NVIDIA GeForce ...  Off  | 00000000:01:00.0 Off |                  N/A |
        # Add Classifier                                                                                          │| 30%   31C    P8    17W / 320W |    128MiB / 10240MiB |      0%      Default |
        self.classifier = nn.Linear(self.model.config.hidden_size, num_classes)  # [768, 10] in CIFAR-10          │|                               |                      |                  N/A |
        self.softmax = nn.Softmax(dim=-1)

    """
    # Without Pretraining
    def __init__(self, num_classes):
        super().__init__()

        # Initializing a ViT vit-base-patch16-224 style configuration
        self.configuration = ViTConfig()

        # Initializing a model (with random weights) from the vit-base-patch16-224 style configuration
        self.model = ViTModel(self.configuration)

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

