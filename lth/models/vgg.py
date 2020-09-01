import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optimizers
from . import utils

cfgs = {
    "A": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "A"],
    "B": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "A"],
    "D": [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        "A",
    ],
    "E": [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        512,
        "A",
    ],
    "F": [64, 64, "M"],
    "G": [64, 64, "M", 128, 128, "M"],
    "H": [64, 64, "M", 128, 128, "M", 256, 256, "M"],
}


def vgg19(batch_norm=False, optim='SGD', lr=0.01, **kwargs):
    return VGG("E", batch_norm, optim=optim, lr=lr, **kwargs)


def conv2(batch_norm=False, optim='Adam', lr=2e-4, **kwargs):
    return VGG("F", batch_norm, optim=optim, lr=lr, **kwargs)


def conv4(batch_norm=False, optim='Adam', lr=3e-4, **kwargs):
    return VGG("G", batch_norm, optim=optim, lr=lr, **kwargs)


def conv6(batch_norm=False, optim='Adam', lr=3e-4, **kwargs):
    return VGG("H", batch_norm, optim=optim, lr=lr, **kwargs)


class VGG(utils.Base):
    def __init__(
        self,
        cfg: str,
        batch_norm: bool,
        optim: str = 'SGD',
        lr: float = 0.01,
        **kwargs
    ):
        super(VGG, self).__init__()
        self.head, self.tail = self._build_model(cfg, batch_norm)

        optimizer = optimizers.__dict__[optim]
        optimizer_kwargs = utils.get_kwargs(optimizer, kwargs)
        self.optim = optimizer(self.parameters(), lr=lr, **optimizer_kwargs)
        self.optim.name = optim

        self._initialize_weights()

    def forward(self, x):
        x = self.head(x)
        x = x.view(x.size(0), -1)
        x = self.tail(x)
        return x

    @staticmethod
    def _build_model(cfg: str, batch_norm: bool):
        head = list()
        fan_in = 3
        cfg = cfgs[cfg]
        for layer in cfg:
            if layer == "M":
                head += [nn.MaxPool2d(kernel_size=2, stride=2)]
            elif layer == "A":
                head += [nn.AvgPool2d(2)]
            else:
                conv2d = nn.Conv2d(fan_in, layer, kernel_size=3, padding=1)
                if batch_norm:
                    head += [conv2d, nn.BatchNorm2d(layer), nn.ReLU(inplace=True)]
                else:
                    head += [conv2d, nn.ReLU(inplace=True)]
                fan_in = layer

        if cfg[-1] == "A":
            tail = [nn.Linear(512, 10)]
        else:
            x = int(64 * 64 * 256/ cfg[-2])
            tail = [nn.Linear(x, 256), nn.Linear(256, 256), nn.Linear(256, 10)]

        return nn.Sequential(*head), nn.Sequential(*tail)
