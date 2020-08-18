import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import CIFAR10, MNIST


def load_CIFAR10(root: str, download: bool = False, augment: bool = False, validation = 5000, **kwargs):

    # data augmentation
    _transform = [
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
    _augment = [transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, 4)]

    transform = transforms.Compose(_augment + _transform if augment else _transform)

    batch_size = kwargs.get("batch_size", 64)

    trainset = CIFAR10(root=root, train=True, download=download, transform=transform)
    trainset, validset = _validation_split(trainset, validation)

    testset = CIFAR10(
        root=root,
        train=False,
        download=download,
        transform=transforms.Compose(_transform),
    )

    trainloader = DataLoader(
        trainset, batch_size=batch_size, shuffle=True
    )

    validloader = DataLoader(
        validset, batch_size=batch_size, shuffle=True
    )

    testloader = DataLoader(
        testset, batch_size=batch_size, shuffle=False,
    ) 
    return trainloader, validloader, testloader


def load_MNIST(root: str, download: bool = False, validation = 5000, **kwargs):

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean=[0.1307], std=[0.3081])]
    )
    batch_size = kwargs.get("batch_size", 64)

    trainset = MNIST(root=root, train=True, download=download, transform=transform)
    trainset, validset = _validation_split(trainset, validation)

    testset = MNIST(root=root, train=False, download=download, transform=transform)

    trainloader = DataLoader(
        trainset, batch_size=batch_size, shuffle=True
    )

    validloader = DataLoader(
        validset, batch_size=batch_size, shuffle=True
    )

    testloader = DataLoader(
        testset, batch_size=batch_size, shuffle=False
    )

    return trainloader, validloader, testloader


def _validation_split(dataloader, validation_size):
    if validation_size < 1:
        split_size = int(validation_size * len(dataloader))
    else:
        split_size = validation_size

    train, validation = random_split(dataloader, [len(dataloader) - split_size, split_size])
    return train, validation


def build_meta(model, data, **kwargs):

    f = {k: v for k, v in kwargs.items()}

    params = model.optim.state_dict()["param_groups"][0]
    params = {k: v for k, v in params.items() if k is not "params"}
    f.update(params)

    f["optimizer"] = str(model.optim.name)
    f["dataset"] = str(data.dataset)
    f["batch_size"] = str(data.batch_size)

    return f


_dispatcher = {"cifar10": load_CIFAR10, "mnist": load_MNIST}
