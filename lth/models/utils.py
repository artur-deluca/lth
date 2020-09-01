import sys
import inspect
import torch
import torch.nn as nn
import torch.nn.functional as F


class Base(nn.Module):
    def __init__(self):
        super(Base, self).__init__()
        self.criterion = nn.CrossEntropyLoss()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


class Residual(Base):
    def __init__(self, f_in: int, f_out: int, downsample: bool):
        super(Residual, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                f_in, f_out, kernel_size=3, stride=downsample + 1, padding=1, bias=False
            ),
            nn.BatchNorm2d(f_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(f_out, f_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(f_out),
        )

        # No parameters for shortcut connections.
        if downsample or f_in != f_out:
            self.shortcut = nn.Sequential(
                nn.Conv2d(f_in, f_out, kernel_size=1, stride=2, bias=False),
                nn.BatchNorm2d(f_out),
            )
        else:
            self.shortcut = nn.Sequential()

    def forward(self, x):
        x = self.block(x) + self.shortcut(x)
        return F.relu(x)


def get_models(name):
    return dict(inspect.getmembers(sys.modules[name], inspect.isfunction))


def get_kwargs(fn, kwargs):
    keys = set(set(inspect.getfullargspec(fn).args) & set(kwargs.keys()))
    return {k: v for k, v in kwargs.items() if k in keys}
