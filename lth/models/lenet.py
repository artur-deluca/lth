from torch import nn
import torch.nn.functional as F
from torchvision.models.vgg import cfgs

from utils import Base

def lenet():
    return LeNet()

class LeNet(Base):
    def __init__(self):
        super(LeNet, self).__init__()
        self.head, self.tail = self._build_layers()

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten.
        x = self.head(x)
        x = self.tail(x)
        return x

    @staticmethod
    def _build_layers():

        head = nn.Sequential(
                nn.Linear(784, 300),
                nn.ReLU(inplace=True),
                nn.Linear(300, 100),
                nn.ReLU(inplace=True)
        )

        tail = nn.Linear(100, 10)
        return head, tail
