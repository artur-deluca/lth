from .resnet import resnet20, resnet32, resnet44, resnet56
from .vgg import vgg19, conv2, conv4, conv6
from .lenet import lenet
from .utils import get_models


_dispatcher = get_models(__name__)
models = _dispatcher.keys()
