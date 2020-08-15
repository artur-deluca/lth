import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optimimzers
from . import utils

def lenet(optim=utils.optim, lr=utils.lr, **kwargs):
    return LeNet()

class LeNet(utils.Base):
    def __init__(self, optim: str = utils.optim, lr: float = utils.lr, **kwargs):
        super(LeNet, self).__init__()
        self.head, self.tail = self._build_layers()
        self.optim = optimimzers.__dict__[optim](self.parameters(), lr=lr, **kwargs)
        self.optim.name = optim
        self._initialize_weights()

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
            nn.ReLU(inplace=True),
        )

        tail = nn.Linear(100, 10)
        return head, tail
