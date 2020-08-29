import csv
import sys
import os
import re
import json
from datetime import datetime

import torch
from loguru import logger


def _set_logger(func):
    def wrapper(*args, **kwargs):
        logger.remove(0)
        message_format = '<level>{level: <8}</level> {message}'
        if os.getenv('verbose'):
            logger.add(sys.stderr, level=os.getenv('verbose'), format=message_format)
        else:
            logger.add(sys.stderr, level='INFO', format=message_format)
        func(*args, **kwargs)
    return wrapper


def _get_device():

    device = None
    if os.getenv('device') and 'cuda' in os.getenv('device'):
        if not torch.cuda.is_available():
            logger.warning('CUDA is not supported in current device, using CPU instead')
        else:
            device = torch.device(os.getenv('device'))
    return device


def _get_save_dir(model, dataset, random):

    save = os.getenv('save_dir')
    dataset = dataset.split('\n')[0].strip('Dataset ')

    if save is None:
        save = './experiments/'

    if save:
        f = f'{datetime.now().strftime("%m_%d_%Y_%I_%M")}_{model}_{dataset}'
        if random:
            f += '_random'

        save = os.path.join(save, f)
        os.makedirs(save, exist_ok=True)
    else:
        logger.warning('Save path set empty, training info will not be stored')

    return save


def _get_seed():
    if os.getenv('seed'):
        return int(os.getenv('seed'))
    return None


def build_meta(model, data, **kwargs):
    
    f = {
        'model': model._get_name(),
        'seed': _get_seed() or '',
        'device': os.getenv('device') or 'cpu',
        'optimizer': str(model.optim.name),
        'dataset': str(data.train.dataset),
        'train_batch_size': data.train.batch_size,
        'train_size': len(data.train) * data.train.batch_size
    }

    f.update(kwargs)
    params = model.optim.state_dict()["param_groups"][0]
    f.update({k: v for k, v in params.items() if k is not 'params'})

    if 'validation' in data: 
        f["validation_batch_size"] = data.validation.batch_size
        f["validation_size"] = len(data.validation) * data.validation.batch_size

    if 'test' in data: 
        f["test_batch_size"] = data.test.batch_size
        f["test_size"] = len(data.test) * data.test.batch_size

    f['round_best_iteration'] = list()
    f['round_test_error'] = list()

    return f

def write_epoch(checkpoint, directory):

    checkpoint = dict(checkpoint._asdict())
    os.makedirs(directory, exist_ok=True)

    with open(os.path.join(directory, "iterations.csv"), 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=checkpoint.keys())

        for i, _ in enumerate(checkpoint['train_loss']):
            writer.writerow({k: v[i] for k, v in checkpoint.items()})

def _fetch_layers(model):

    loc = locals()
    params = [name for name, _ in model.named_parameters()]
    params = [re.sub(r"(\.)([0-9]+)", r"[\2]", name.split('_')[0]) for name in params]
    params = [
        (eval("model." + attr, loc), param_type)
        for (attr, _, param_type, _) in [
            re.split(r"(\.)(\w+$)", name) for name in params
        ]
    ]

    return params


def sparsity(model, globally=False):

    params = [getattr(layer, name) for layer, name in _fetch_layers(model)]
    if globally:
        return 100.0 * sum([int(torch.sum(x == 0)) for x in params]) / sum([x.nelement() for x in params])

    return 100.0 * (sum([int(torch.sum(x == 0)) / x.nelement() for x in params]) / len(params))
