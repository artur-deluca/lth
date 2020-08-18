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
    device = None
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
        model.load_state_dict(original_weights)

        for epoch in range(1, epochs + 1):

            losses["train"].append(evaluate(trainloader, train=True))
            losses["validation"].append(evaluate(validloader, train=False))
            losses["test"].append(evaluate(testloader, train=False))

            print(
                    f"[round: {r} | epoch: {epoch}] train: {losses['train'][-1]:.3f} validation: {losses['validation'][-1]:.3f}"
            ) if verbose else None

            if earlystopping:
                if abs(losses['validation'][-1] - prev_loss) / prev_loss < tol:
                    streak += 0
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


def prune_net(model, rate, fc_rate=None, globally=False):
    loc = locals()
    params = [name for name, _ in model.named_parameters()]
    params = [re.sub(r"(\.)([0-9]+)", r"[\2]", name) for name in params]
    params = [
        (eval("model." + attr, loc), param_type)
        for (attr, _, param_type, _) in [
            re.split(r"(\.)(\w+$)", name) for name in params
        ]
    ]

    if globally:
        prune.global_unstructured(
            params, amount=rate, pruning_method=prune.L1Unstructured
        )
        for (layer, param_type) in params:
            prune.remove(layer, param_type)
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
        prune.remove(layer, param_type)

    return model
