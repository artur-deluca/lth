from torch import nn
import torch.nn.functional as F

from .utils import Base, Residual

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
        self.head, self.body, self.tail = self._build_body(depth, width)
        self._initialize_weights()

    def forward(self, x):

        x = self.head()
        x = self.body(x)
        x = F.avg_pool2d(x, x.size()[3])
        x = x.view(x.size(0), -1)
        x = self.tail(x)

        return x

    @staticmethod
    def _build_layers(depth: int, width:int):

        head = nn.Sequential(
                nn.Conv2d(3, width, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(width),
                nn.ReLU(inplace=True)
        )

        body = list()
        fan_in = width
        for block_id in [1, 2, 4]:

            fan_out = width * block_id
            for layer_id in range(depth):
                downsample = block_id > 1 and layer_id == 0
                body.append(Residual(fan_in, fan_out, downsample))
                fan_in = fan_out
        body = nn.Sequential(*body)

        tail = nn.Linear(block_id * width, 10) 

        return head, body, tail 


