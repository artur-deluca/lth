from collections import namedtuple

from torch import nn
import torch.nn.utils.prune as prune

try:
    import utils
except ImportError:
    from . import utils

prune_container = namedtuple('prune_container', 'rate, fc_rate, globally')

def prune_net(model, rate, fc_rate=None, globally=False):
    
    params = utils._fetch_layers(model)

    if globally:
        params = [(layer, name) for layer, name in params if not isinstance(layer, nn.Linear)]
        prune.global_unstructured(
            params, amount=rate, pruning_method=prune.L1Unstructured
        )
        return model

    if str(fc_rate).isnumeric():
        rates = [
            fc_rate if isinstance(layer, nn.Linear) else rate
            for layer, _ in params
        ]
    else:
        rates = [rate for _ in range(len(params))]

    for (layer, param_type), rate in zip(params, rates):
        prune.l1_unstructured(layer, name=param_type, amount=rate)

    return model
