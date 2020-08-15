import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10, MNIST


def load_CIFAR10(root: str, download: bool = False, augment: bool = False, **kwargs):

    # data augmentation
    _transform = [
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
    _augment = [transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, 4)]

    transform = transforms.Compose(_augment + _transform if augment else _transform)

    batch_size = kwargs.get("batch_size", 4)
    num_workers = kwargs.get("num_workers", 2)

    trainset = CIFAR10(root=root, train=True, download=download, transform=transform)
    trainloader = DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    testset = CIFAR10(
        root=root,
        train=False,
        download=download,
        transform=transforms.Compose(_transform),
    )
    testloader = DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return trainloader, testloader


def load_MNIST(root: str, download: bool = False, **kwargs):

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean=[0.1307], std=[0.3081])]
    )
    batch_size = kwargs.get("batch_size", 4)
    num_workers = kwargs.get("num_workers", 2)

    trainset = MNIST(root=root, train=True, download=download, transform=transform)
    trainloader = DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    testset = MNIST(root=root, train=False, download=download, transform=transform)
    testloader = DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return trainloader, testloader


def build_meta(model, data, **kwargs):

    f = {k: v for k, v in kwargs.items()}

    params = model.optim.state_dict()['param_groups'][0]
    params = {k:v for k,v in params.items() if k is not 'params'}
    f.update(params)

    f["optimizer"] = str(model.optim.name)
    f["dataset"] = str(data.dataset)
    f["batch_size"] = str(data.batch_size)

    return f

_dispatcher = {"cifar10": load_CIFAR10, "mnist": load_MNIST}
