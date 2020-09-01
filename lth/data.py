from collections import namedtuple

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import CIFAR10, MNIST

_dataloader = namedtuple('dataloader', ['train', 'validation', 'test'])

def load_CIFAR10(root: str, download: bool = False, augment: bool = False, validation = 5000, **kwargs):

    # data augmentation
    _transform = [
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
    _augment = [transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, 4)]

    transform = transforms.Compose(_augment + _transform if augment else _transform)
    train_batch_size = kwargs.get("batch_size", 128)
    valid_batch_size = kwargs.get("validation_batch_size", len(validset) // 10)
    test_batch_size = kwargs.get("test_batch_size", len(testset) // 10)

    trainset = CIFAR10(root=root, train=True, download=download, transform=transform)
    trainset, validset = _validation_split(trainset, validation)

    testset = CIFAR10(
        root=root,
        train=False,
        download=download,
        transform=transforms.Compose(_transform),
    )

    trainloader = DataLoader(
        trainset, batch_size=train_batch_size, shuffle=True, num_workers=8, pin_memory=True
    )

    validloader = DataLoader(
        validset, batch_size=valid_batch_size, shuffle=True, num_workers=8, pin_memory=True
    )

    testloader = DataLoader(
        testset, batch_size=test_batch_size, shuffle=False, num_workers=8, pin_memory=True
    )

    return _dataloader(train=trainloader, validation=validloader, test=testloader)


def load_MNIST(root: str, download: bool = False, validation = 5000, **kwargs):

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean=[0.1307], std=[0.3081])]
    )

    trainset = MNIST(root=root, train=True, download=download, transform=transform)
    trainset, validset = _validation_split(trainset, validation)

    testset = MNIST(root=root, train=False, download=download, transform=transform)

    train_batch_size = kwargs.get("batch_size", 60)
    valid_batch_size = kwargs.get("validation_batch_size", len(validset) // 5)
    test_batch_size = kwargs.get("test_batch_size", len(testset) // 5)

    trainloader = DataLoader(
        trainset, batch_size=train_batch_size, shuffle=True, num_workers=8, pin_memory=True
    )

    validloader = DataLoader(
        validset, batch_size=valid_batch_size, shuffle=True, num_workers=8, pin_memory=True
    )

    testloader = DataLoader(
        testset, batch_size=test_batch_size, shuffle=False, num_workers=8, pin_memory=True
    )

    return _dataloader(train=trainloader, validation=validloader, test=testloader)


def _validation_split(dataloader, validation_size):
    if validation_size < 1:
        split_size = int(validation_size * len(dataloader))
    else:
        split_size = validation_size

    train, validation = random_split(dataloader, [len(dataloader) - split_size, split_size])
    return train, validation


_dispatcher = {"cifar10": load_CIFAR10, "mnist": load_MNIST}
