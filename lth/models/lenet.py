import torch.optim as optimizers
import torch.nn as nn
import torch.nn.functional as F
from . import utils

def lenet(optim='Adam', lr=1.2e-3, **kwargs):
    return LeNet(optim, lr, **kwargs)

class LeNet(utils.Base):
    def __init__(self, optim: str, lr: float, **kwargs):
        super(LeNet, self).__init__()
        self.head, self.tail = self._build_layers()

        optimizer = optimizers.__dict__[optim]
        optimizer_kwargs = utils.get_kwargs(optimizer, kwargs)
        self.optim = optimizer(self.parameters(), lr=lr, **optimizer_kwargs)
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
