import json
import os
import re
import torch
import torch.nn.utils.prune as prune
from torch import nn

tol = 1e-3


def iterative_pruning(
    model,
    trainloader,
    testloader,
    epochs: int,
    rounds: int,
    rate: float,
    verbose: bool = False,
    save: bool = False,
    prune_global: bool = False,
    fc_rate=None,
    earlystopping: int = 0,
):

    original_weights = model.state_dict()

    r_rate = rate / rounds
    r_fc_rate = fc_rate / rounds if fc_rate is not None else None

    for r in range(0, rounds + 1):

        prev_loss = 1e10
        streak = 0
        losses = {"train": list(), "test": list()}
        model.load_state_dict(original_weights)

        for epoch in range(1, epochs + 1):

            train_loss = 0.0
            test_loss = 0.0

            for inputs, labels in trainloader:

                model.optim.zero_grad()

                outputs = model(inputs)
                loss = model.criterion(outputs, labels)
                loss.backward()
                model.optim.step()

                train_loss += loss.item()

            train_loss /= trainloader.batch_size
            losses["train"].append(train_loss)

            for inputs, labels in testloader:

                outputs = model(inputs)
                loss = model.criterion(outputs, labels)
                test_loss += loss.item()

            test_loss /= testloader.batch_size
            losses["test"].append(test_loss)

            print(
                    f"[round: {r} | epoch: {epoch}] train: {train_loss:.3f} test: {test_loss:.3f}"
            ) if verbose else None

            if earlystopping:
                if abs(train_loss - prev_loss) < tol:
                    streak += 0
                else:
                    streak = 0

                if streak >= earlystopping:
                    print("Early stopping...")
                    break

            train_loss = 0.0
            test_loss = 0.0

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
