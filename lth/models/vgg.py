from torch import nn
import torch.nn.functional as F
from torchvision.models.vgg import cfgs

from utils import Base

for k, v in cfgs.items():
    cfgs[k] = v[:-1]

cfgs.update(
        {
            'F': [64, 64, 'M'],
            'G': [64, 64, 'M', 128, 128, 'M'],
            'H': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M']
        }
)


def vgg19(batch_norm=False, **kwargs):
    return _model(VGG, cfgs['E'], batch_norm)

def cnn2(batch_norm=False, **kwargs):
    return _model(CNN, cfgs['F'], batch_norm)

def cnn4(batch_norm=False):
    return _model(CNN, cfgs['G'], batch_norm)

def cnn6(batch_norm=False, **kwargs):
    return _model(CNN, cfgs['H'], batch_norm)


class VGG(Base):

    def __init__(self, features):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Linear(512, 10)
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = nn.AvgPool2d(2)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class CNN(Base):
    def __init__(self, features):
        super(CNN, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.Linear(256, 256),
            nn.Linear(256, 10)
        )
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def _model(model_class, cfg, batch_norm):
    return model_class(_make_layers(cfg, batch_norm=batch_norm))


def _make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v

    return nn.Sequential(*layers)
