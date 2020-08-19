import json
import os
import re
import torch
import torch.nn.utils.prune as prune
from torch import nn

tol = 0.01

def iterative_pruning(
    model,
    trainloader,
    validloader,
    testloader,
    epochs: int,
    rounds: int,
    rate: float,
    verbose: bool = False,
    save: bool = False,
    prune_global: bool = False,
    fc_rate=None,
    earlystopping: int = 0,
    device = None,
    **kwargs
):
    if device:
        model = model.to(device)
        print(model.head[0].weight.type())

    original_weights = model.state_dict()

    def evaluate(dataloader, train=False):
        dataloss = 0.0
        if train:
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

                outputs = model(inputs)
                loss = model.criterion(outputs, labels)
                dataloss += loss.item()

        return dataloss / dataloader.batch_size


    for r in range(0, rounds + 1):

        prev_loss = 1e10
        streak = 0
        losses = {'train': list(), 'test': list(), 'validation': list()}

        placeholder = model.state_dict()
        keys = [label for label in placeholder.keys() if not label.endswith("_mask")]
        for k in original_weights.keys():
            placeholder[[x for x in keys if x.startswith(k)][0]] = original_weights[k]

        model.load_state_dict(placeholder)
            
        for epoch in range(1, epochs + 1):

            losses["train"].append(evaluate(trainloader, train=True))
            losses["validation"].append(evaluate(validloader, train=False))
            losses["test"].append(evaluate(testloader, train=False))

            sparsity = get_sparsity(model, prune_global)
            print(
                    f"[round: {r} | epoch: {epoch}] train: {losses['train'][-1]:.3f} validation: {losses['validation'][-1]:.3f} | sparsity: {sparsity:.0f}"
            ) if verbose else None

            if earlystopping:
                if abs(losses['validation'][-1] - prev_loss) / prev_loss < tol:
                    streak += 1
                else:
                    streak = 0

                if streak >= earlystopping:
                    print("Early stopping...")
                    break

            prev_loss = losses['validation'][-1] 

        if save:
            directory = os.path.join(save, str(r))
            write_data(model, losses, directory)

        if r != rounds + 1:
            model = prune_net(model, rate, fc_rate, prune_global)

    print("Finished Training")


def write_data(model, losses, directory):
    os.makedirs(directory, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(directory, "weights"))
    with open(os.path.join(directory, "loss.json"), "w") as f:
        json.dump(losses, f)

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

def get_sparsity(model, globally=False):

    buffers = [x for name, x in model.named_buffers() if 'mask' in name]
    
    try:
        if globally:
            return 100.0 * sum([int(torch.sum(x == 0)) for x in buffers]) / sum([x.nelement() for x in buffers])

        return 100.0 * (sum([int(torch.sum(x == 0)) / x.nelement() for x in buffers]) / len(buffers))

    except ZeroDivisionError:
        return 0

def prune_net(model, rate, fc_rate=None, globally=False):

    params = _fetch_layers(model)

    if globally:
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

