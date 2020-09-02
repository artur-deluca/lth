import csv
import numbers
import sys
import os
import re
import json
from datetime import datetime
from inspect import signature

import torch
from loguru import logger


def _logger(func):
    def wrapper(*args, **kwargs):
        logger.remove(0)
        message_format = "<level>{level: <8}</level> {message}"
        if os.getenv("verbose"):
            logger.add(sys.stderr, level=os.getenv("verbose"), format=message_format)
        else:
            logger.add(sys.stderr, level="INFO", format=message_format)
        func(*args, **kwargs)

    return wrapper


def set_params(**kwargs):
    if "seed" in kwargs:
        set_seed(kwargs["seed"])

    if "eval_step" in kwargs:
        set_eval_step(kwargs["eval_step"])

    if "device" in kwargs:
        set_device(kwargs["device"])

    if "verbosity" in kwargs:
        set_verbosity(kwargs["verbose"])


def set_seed(seed):
    if not isinstance(seed, numbers.Number):
        raise TypeError("Seed is not a number")
    else:
        os.environ["seed"] = str(seed)
        torch.manual_seed(int(seed))


def set_eval_step(step):
    if not isinstance(step, numbers.Number):
        raise TypeError("Step is not a number")
    os.environ["eval_step"] = str(step)


def set_verbosity(verbosity):
    os.environ["verbosity"] = verbosity


def set_device(device):
    os.environ["device"] = device


def _get_eval_step(train):
    step = int(os.getenv("eval_step") or 1)

    if step <= 0:
        return len(train)

    return step


def _get_prune_meta(func):
    meta = {'fn_name': _get_fn_name(func)}
    args = signature(func).parameters.values()
    meta['fn_args'] = ', '.join(str(x) for x in args)

    return meta


def _get_fn_name(func):
    try:
        name = func.__name__
        return name

    except AttributeError:
        try:
            name = func.func.__name__
            return name

        except AttributeError:
            return None


def _get_device():

    device = None
    if os.getenv("device") and "cuda" in os.getenv("device"):
        if not torch.cuda.is_available():
            logger.warning("CUDA is not supported in current device, using CPU instead")
        else:
            device = torch.device(os.getenv("device"))
            torch.backends.cudnn.benchmark = True

    return device


def _get_save_dir(model, dataset, random):

    save = os.getenv("save_dir")
    dataset = dataset.split("\n")[0].strip("Dataset ")

    if save is None:
        save = "./experiments/"

    if save:
        f = f'{datetime.now().strftime("%m_%d_%Y_%I_%M")}_{model}_{dataset}'
        if random:
            f += "_random"

        save = os.path.join(save, f)
        os.makedirs(save, exist_ok=True)
    else:
        logger.warning("Save path set empty, training info will not be stored")

    return save


def _get_seed():
    if os.getenv("seed"):
        return int(os.getenv("seed"))
    return None


def build_meta(model, data, **kwargs):

    dataset = str(data.train.dataset.dataset)
    dataset = dataset.split("\n")[0].strip("Dataset ")

    f = {
        "model": model._get_name(),
        "seed": _get_seed() or "",
        "device": os.getenv("device") or "cpu",
        "optimizer": str(model.optim.name),
        "dataset": dataset,
        "train_batch_size": data.train.batch_size,
        "train_size": len(data.train) * data.train.batch_size,
    }

    f.update(kwargs)
    params = model.optim.state_dict()["param_groups"][0]
    f.update({k: v for k, v in params.items() if k is not "params"})

    if "validation" in data:
        f["validation_batch_size"] = data.validation.batch_size
        f["validation_size"] = len(data.validation) * data.validation.batch_size

    if "test" in data:
        f["test_batch_size"] = data.test.batch_size
        f["test_size"] = len(data.test) * data.test.batch_size

    f["round_best_iteration"] = list()
    f["round_test_error"] = list()

    return f


def write_epoch(checkpoint, directory):

    checkpoint = dict(checkpoint._asdict())
    os.makedirs(directory, exist_ok=True)
    f = os.path.join(directory, "iterations.csv")
    file_exists = os.path.isfile(f)

    with open(f, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=checkpoint.keys())

        if not file_exists:
            writer.writeheader()

        for i, _ in enumerate(checkpoint["train_loss"]):
            writer.writerow({k: v[i] for k, v in checkpoint.items()})


