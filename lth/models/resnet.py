from torch import nn
import torch.nn.functional as F

from utils import Base

def resnet20():
    return ResNet(6)

def resnet32():
    return ResNet(10)

def resnet44():
    return ResNet(14)

def resnet56():
    return ResNet(18)


class ResNet(Base):

    def __init__(self, depth: int, width: int = 16):
        super(ResNet, self).__init__()

        self.head = nn.Sequential(
                nn.Conv2d(3, width, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(16),
                nn.ReLU()
        )

        features = list()
        fan_in = width
        for block_id in [1, 2, 4]:

            fan_out = width * block_id
            for layer_id in range(depth):
                downsample = block_id > 1 and layer_id == 0
                features.append(Residual(fan_in, fan_out, downsample))
                fan_in = fan_out

        self.features = nn.Sequential(*features)
        self.tail = nn.Linear(block_id * width, 10) 

        self._initialize_weights()

    def forward(self, x):

        x = self.head()
        x = self.features(x)
        x = F.avg_pool2d(x, x.size()[3])
        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        return x

# TODO: see if we need to add Base here
class Residual(nn.Module):

    def __init__(self, f_in: int, f_out: int, downsample: bool):
        super(Model.Block, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(f_in, f_out, kernel_size=3, stride=downsample + 1, padding=1, bias=False),
            nn.BatchNorm2d(f_out),
            nn.ReLU(),
            nn.Conv2d(f_out, f_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(f_out)
        )

        # No parameters for shortcut connections.
        if downsample or f_in != f_out:
            self.shortcut = nn.Sequential(
                nn.Conv2d(f_in, f_out, kernel_size=1, stride=2, bias=False),
                nn.BatchNorm2d(f_out)
            )
        else:
            self.shortcut = nn.Sequential()

    def forward(self, x):
        x = self.block(x) + self.shortcut(x)
        return F.relu(x)
