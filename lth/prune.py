import re
from collections import namedtuple

import torch
import torch.nn.utils.prune as prune
from torch import nn

try:
    import utils
except ImportError:
    from . import utils

prune_container = namedtuple("prune_container", "rate, fc_rate, globally")

def fetch_layers(model):
    """Fetch all parameters as tuples according to `torch.nn.prune` format
    Args
        model: nn.Module
    Returns:
        params: list of tuples with layer and name of parameter
        (e.g. (nn.Linear(...), 'weight'))
    """

    loc = locals()
    params = [name for name, _ in model.named_parameters()]
    params = [re.sub(r"(\.)([0-9]+)", r"[\2]", name.split("_")[0]) for name in params]
    params = [
        (eval("model." + attr, loc), param_type)
        for (attr, _, param_type, _) in [
            re.split(r"(\.)(\w+$)", name) for name in params
        ]
    ]

    return params


def fetch_buffers(model):
    """Fetch all named buffers"""

    loc = locals()
    buffers = [name for name, _ in model.named_buffers()]
    buffers = [re.sub(r"(\.)([0-9]+)", r"[\2]", name.split("_")[0]) for name in buffers]
    buffers = [
        (eval("model." + attr, loc), param_type)
        for (attr, _, param_type, _) in [
            re.split(r"(\.)(\w+$)", name) for name in buffers
        ]
    ]

    return buffers



def sparsity(model):
    """Get model sparsity in percentage.
    Sparsity is here calculated as the layer-wise average of null items across the network
    """
    
    buffers = fetch_buffers(model)

    params = [getattr(layer, name) for layer, name in fetch_layers(model)]
    sparsity_layers = [int(torch.sum(x == 0)) for x in params]
    size_layers = [int(x.nelement()) for x in params]

    return 100.0 * sum(sparsity_layers) / sum(size_layers)


def prune_all(model, rate, prune_method=prune.l1_unstructured):

    layers = fetch_layers(model)
    for (layer, param_type) in layers[:-2]:
        prune_method(layer, name=param_type, amount=rate)

    for (layer, param_type) in layers[-2:]:
        prune_method(layer, name=param_type, amount=rate / 2)

    return model

def prune_conv_globally(model, rate, pruning_class=prune.L1Unstructured):

    params = fetch_layers(model)
    params = [(layer, name) for layer, name in params if not isinstance(layer, nn.Linear)]

    prune.global_unstructured(params, amount=rate, pruning_method=pruning_class)

    return model

def prune_fc(model, rate, prune_method=prune.l1_unstructured):

    params = fetch_layers(model)
    params = [(layer, name) for layer, name in params if isinstance(layer, nn.Linear)]

    for (layer, param_type) in params[:-2]:
        prune_method(layer, name=param_type, amount=rate)

    for (layer, param_type) in params[-2:]:
        prune_method(layer, name=param_type, amount=rate / 2)

    return model

def prune_conv(model, rate, prune_method=prune.l1_unstructured):

    params = fetch_layers(model)
    params = [(layer, name) for layer, name in params if not isinstance(layer, nn.Linear)]

    for (layer, param_type) in params:
        prune_method(layer, name=param_type, amount=rate)

    return model

