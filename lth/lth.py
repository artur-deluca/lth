import math
import json
import os
import re
from copy import deepcopy

import torch
import torch.nn.utils.prune as prune
from torch import nn


def iterative_pruning(
    model,
    trainloader,
    validloader,
    testloader,
    iterations: int,
    rounds: int,
    rate: float,
    verbose: bool = False,
    save: bool = False,
    prune_global: bool = False,
    fc_rate=None,
    device = None,
    **kwargs
):
    if device:
        model = model.to(device)

    original_weights = deepcopy(model.state_dict())
    best_model = dict()

    def evaluate(dataloader, trainable=False):
        dataloss = 0.0
        if trainable:
            for inputs, labels in dataloader:

                if device: inputs, labels = inputs.to(device), labels.to(device)

                model.optim.zero_grad()

                outputs = model(inputs)
                loss = model.criterion(outputs, labels)
                loss.backward()
                model.optim.step()

                dataloss += loss.item()
        else:

            for inputs, labels in dataloader:

                if device: inputs, labels = inputs.to(device), labels.to(device)

                with torch.no_grad():

                    outputs = model(inputs)
                    loss = model.criterion(outputs, labels)
                    dataloss += loss.item()

        return dataloss / len(dataloader)


    for r in range(0, rounds + 1):

        min_loss = 1e10
        summary = {'train': list(), 'validation': list(), 'sparsity': 0}

        placeholder = deepcopy(model.state_dict())
        keys = [label for label in placeholder.keys() if not label.endswith("_mask")]
        for k in original_weights.keys():
            placeholder[[x for x in keys if x.startswith(k)][0]] = original_weights[k]

        model.load_state_dict(placeholder)
        epochs = math.ceil(iterations / len(trainloader))
            
        for epoch in range(1, epochs + 1):

            summary["train"].append(evaluate(trainloader, trainable=True))
            summary["validation"].append(evaluate(validloader, trainable=False))
            summary['sparsity'] = sparsity(model, prune_global)

            if verbose:
                message = f"[round: {r} | epoch: {epoch}] "
                message += f"train: {summary['train'][-1]:.3f} validation: {summary['validation'][-1]:.3f} "
                message += f"| sparsity: {summary['sparsity']:.0f}%"
                print(message)

            if summary['validation'][-1] < min_loss:
                min_loss = summary['validation'][-1]
                best_model = deepcopy(dict(model.named_parameters()))
                best_model.update(deepcopy(dict(model.named_buffers())))
        
        model.load_state_dict(best_model)
        summary['best_iteration'] = summary['validation'].index(min_loss)
        summary['test_error'] = evaluate(testloader, trainable=False)

        if save:
            directory = os.path.join(save, str(r))
            write_data(best_model, summary, directory)

        if r != rounds + 1:
            model = prune_net(model, rate, fc_rate, prune_global)

    print("Finished Training")


def write_data(state, summary, directory):

    os.makedirs(directory, exist_ok=True)
    torch.save(state, os.path.join(directory, "weights"))
    with open(os.path.join(directory, "loss.json"), "w") as f:
        json.dump(summary, f)

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


def prune_net(model, rate, fc_rate=None, globally=False):
    
    params = _fetch_layers(model)

    if globally:
        params = [(layer, name) for layer, name in params if not isinstance(layer, nn.Linear)]
        prune.global_unstructured(
            params, amount=rate, pruning_method=prune.L1Unstructured
        )
        return model

    if str(fc_rate).isnumeric():
        rates = [
            fc_rate if isinstance(layer, torch.nn.Linear) else rate
            for layer, _ in params
        ]
    else:
        rates = [rate for _ in range(len(params))]

    for (layer, param_type), rate in zip(params, rates):
        prune.l1_unstructured(layer, name=param_type, amount=rate)

    return model

