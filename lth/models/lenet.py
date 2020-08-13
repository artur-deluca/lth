from torch import nn
import torch.nn.functional as F
from torchvision.models.vgg import cfgs

from utils import Model

class LeNet(Base):
    def __init__(self):
        super(LeNet, self).__init__()
        self.classifier = nn.Sequential(
                nn.Linear(784, 300),
                nn.Linear(300, 100),
                nn.Linear(100, 10)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten.
        for layer in self.classifier[:-1]:
           x = F.relu(layer(x))

        return self.classifier[-1](x)
