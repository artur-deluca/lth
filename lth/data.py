from collections import namedtuple

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import CIFAR10, MNIST

datawrapper = namedtuple("dataloader", ["train", "validation", "test"])

# TODO: find better batch_size defaults
def load_CIFAR10(
    root: str, augment: bool = False, validation=5000, **kwargs
):
    """Get CIFAR10 datawrapper
    Args:
        root: str
            Path to folder where the dataset is or should be stored
        augment: bool
            Apply pre-defined augmentation
        validation: int or float, default 5000
            Split train and validation dataset.
            If `validation < 1`, split will be done as a portion of the dataset.
            Otherwise, the given number will correspond to the number of observations of the validation set.

    kwargs:
        train_batch_size: int, default 128
        validation_batch_size: int, default 500,
        test_batch_size: int, default 500
        
    Returns:
        datawrapper: namedtuple
            containing 'train', 'validation' and 'test' dataloader.

    """
    # data transform
    _transform = [
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
    # data augmentation
    _augment = [transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, 4)]
    train_transform = transforms.Compose(_augment + _transform if augment else _transform)
    test_transform = transforms.Compose(_transform)

    try:
        trainset = CIFAR10(root=root, train=True, transform=train_transform)
        testset = CIFAR10(root=root, train=False, transform=test_transform)

    # with data is not yet downloaded
    except RuntimeError as err:
        while "invalid answer":

            question = f'No dataset was found in {root}. Would you like to download it? [Y/n]: '
            reply = str(input(question)).lower().strip()
            if not reply or reply[0] == 'y':

                trainset = CIFAR10(root=root, train=True, download=True, transform=train_transform)
                testset = CIFAR10(root=root, train=False, download=True, transform=test_transform)
                break

            elif reply[0] == 'n':
                raise err

    validloader = None
    if validation:
        trainset, validset = _validation_split(trainset, validation)
        valid_batch_size = kwargs.get("validation_batch_size", 500)

        validloader = DataLoader(
            validset,
            batch_size=valid_batch_size,
            shuffle=True,
            num_workers=8,
            pin_memory=True,
        )

    train_batch_size = kwargs.get("train_batch_size", 128)
    trainloader = DataLoader(
        trainset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
    )
    
    test_batch_size = kwargs.get("test_batch_size", 500)
    testloader = DataLoader(
        testset,
        batch_size=test_batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
    )

    return datawrapper(train=trainloader, validation=validloader, test=testloader)


def load_MNIST(root: str, validation=5000, **kwargs):
    """Get MNIST datawrapper
    Args:
        root: str
            Path to folder where the dataset is or should be stored
        augment: bool
            Apply pre-defined augmentation
        validation: int or float, default 5000
            Split train and validation dataset.
            If `validation < 1`, split will be done as a portion of the dataset.
            Otherwise, the given number will correspond to the number of observations of the validation set.

    kwargs:
        train_batch_size: int, default 60
        validation_batch_size: int, default 500,
        test_batch_size: int, default 500 
        
    Returns:
        datawrapper: namedtuple
            containing 'train', 'validation' and 'test' dataloader.

    """
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean=[0.1307], std=[0.3081])]
    )

    try:
        trainset = MNIST(root=root, train=True, transform=transform)
        testset = MNIST(root=root, train=False, transform=transform)

    # with data is not yet downloaded
    except RuntimeError as err:
        while "invalid answer":

            question = f'No dataset was found in {root}. Would you like to download it? [Y/n]: '
            reply = str(input(question)).lower().strip()
            if not reply or reply[0] == 'y':
                trainset = MNIST(root=root, train=True, download=True, transform=transform)
                testset = MNIST(root=root, train=False, download=True, transform=transform)
                break

            elif reply[0] == 'n':
                raise err

    validloader = None
    if validation:
        trainset, validset = _validation_split(trainset, validation)
        valid_batch_size = kwargs.get("validation_batch_size", 500)
        validloader = DataLoader(
            validset,
            batch_size=valid_batch_size,
            shuffle=True,
            num_workers=8,
            pin_memory=True,
        )

    train_batch_size = kwargs.get("train_batch_size", 60)
    trainloader = DataLoader(
        trainset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
    )

    test_batch_size = kwargs.get("test_batch_size", 500)
    testloader = DataLoader(
        testset,
        batch_size=test_batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
    )

    return datawrapper(train=trainloader, validation=validloader, test=testloader)


def _validation_split(dataloader, validation_size):
    if validation_size < 1:
        split_size = int(validation_size * len(dataloader))
    else:
        split_size = validation_size

    train, validation = random_split(
        dataloader, [len(dataloader) - split_size, split_size]
    )
    return train, validation


_dispatcher = {"cifar10": load_CIFAR10, "mnist": load_MNIST}
