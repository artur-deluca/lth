import math
import os
import json
from copy import deepcopy
from collections import namedtuple

import torch
from loguru import logger

try:
    import utils
    from prune import prune_net

except ImportError:
    from . import utils
    from .prune import prune_net


@logger.catch
@utils._logger
def iterative_pruning(
    model, data, iterations: int, rounds: int, prune, random=False, **kwargs
):

    save = utils._get_save_dir(
        model._get_name(), str(data.train.dataset.dataset), random
    )
    logger.debug(f"save path: {save}")

    device = utils._get_device()
    if device:
        model = model.to(device)

    step = utils._get_eval_step(data.train)

    meta = utils.build_meta(
        model,
        data,
        iterations=iterations,
        rounds=rounds,
        rate=prune.rate,
        prune_global=prune.globally,
        fc_rate=prune.fc_rate,
        step=step,
        random=random,
    )

    if random:
        original_weights = deepcopy(model)
        original_weights._initialize_weights()
        original_weights = deepcopy(original_weights.state_dict())
    else:
        original_weights = deepcopy(model.state_dict())

    def evaluate(dataloader):
        dataloss, correct = 0.0, 0.0
        for inputs, labels in dataloader:

            if device:
                inputs, labels = inputs.to(device), labels.to(device)

            with torch.no_grad():

                outputs = model(inputs)
                loss = model.criterion(outputs, labels)
                dataloss += loss.item()

            predictions = torch.argmax(outputs, dim=1)
            correct += (predictions == labels).float().sum()

        accuracy = (correct / (len(dataloader) * dataloader.batch_size)).item()
        loss = dataloss / len(dataloader)

        return loss, accuracy


    template = namedtuple(
        "checkpoint",
        "iteration, train_loss, sparsity, valid_loss, valid_acc, test_loss, test_acc",
    )

    cp = template(
        iteration=list(),
        train_loss=list(),
        sparsity=0.0,
        valid_loss=list(),
        valid_acc=list(),
        test_loss=list(),
        test_acc=list(),
    )  # checkpoint

    for r in range(0, rounds + 1):

        min_loss = 1e10

        placeholder = deepcopy(model.state_dict())
        keys = [label for label in placeholder.keys() if not label.endswith("_mask")]
        for k in original_weights.keys():
            placeholder[[x for x in keys if x.startswith(k)][0]] = original_weights[k]

        model.load_state_dict(placeholder)

        i = 0

        best_model = dict()

        while i < iterations:
            for inputs, labels in data.train:

                cp.iteration.append(i + 1)
                if device:
                    inputs, labels = inputs.to(device), labels.to(device)

                model.optim.zero_grad()

                outputs = model(inputs)
                loss = model.criterion(outputs, labels)
                loss.backward()
                model.optim.step()
                cp.train_loss.append(loss.item())

                if (i + 1) % step == 0:

                    if data.test:

                        test_loss, test_acc = evaluate(data.test)
                        cp.test_loss.append(test_loss)
                        cp.test_acc.append(test_acc)

                    if data.validation:

                        valid_loss, valid_acc = evaluate(data.validation)
                        cp.valid_loss.append(valid_loss)
                        cp.valid_acc.append(valid_acc)

                        if valid_loss < min_loss:
                            best_iter = i
                            min_loss = valid_loss
                            best_model = deepcopy(dict(model.named_parameters()))
                            best_model.update(deepcopy(dict(model.named_buffers())))
                            best_test_loss = test_loss if data.test else None

                else:

                    if data.test:
                        cp.test_loss.append(float("nan"))
                        cp.test_acc.append(float("nan"))

                    if data.validation:
                        cp.valid_loss.append(float("nan"))
                        cp.valid_acc.append(float("nan"))

                if (i + 1) % len(data.train) == 0:

                    epoch = (i + 1) // len(data.train)
                    loss = sum(cp.train_loss) / len(data.train)

                    message = f"[round: {r} | epoch: {epoch}] "
                    message += f"train: {loss:.3f} "
                    if cp.valid_loss:
                        message += f"validation: {cp.valid_loss[-1]:.4f} "
                    message += f"| sparsity: {cp.sparsity:.0f}%"

                    logger.info(message)

                    if save:

                        cp = cp._replace(sparsity=[cp.sparsity for _ in cp.train_loss])
                        utils.write_epoch(cp, os.path.join(save, str(r)))

                    cp = template(
                        iteration=list(),
                        train_loss=list(),
                        sparsity=0.0,
                        valid_loss=list(),
                        valid_acc=list(),
                        test_loss=list(),
                        test_acc=list(),
                    )  # checkpoint

                i += 1

        model.load_state_dict(best_model)

        if data.validation:
            meta["round_best_iteration"].append(best_iter)
            if data.test:
                meta["round_test_error"].append(best_test_loss)

            if save:
                torch.save(best_model, os.path.join(save, f"{r}/weights.pth"))
                with open(os.path.join(save, "meta.json"), "w") as f:
                    json.dump(meta, f)

        if r < rounds:
            model = prune_net(model, prune.rate, prune.fc_rate, prune.globally)
            cp = cp._replace(sparsity=utils.sparsity(model, prune.globally))

    logger.success("Finished pruning")

    return model
