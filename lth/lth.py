import math
import os
import json
import time
from copy import deepcopy
from collections import namedtuple
from typing import Callable, Union

import torch
from loguru import logger
from torch.utils.data import DataLoader

try:
    import utils
    import prune

except ImportError:
    from . import utils
    from . import prune

Num = Union[int, float]


@logger.catch
@utils._logger
def iterative_pruning(
    model, data, iterations: int, rounds: int, prune_net: Callable, **kwargs
):
    """Iterative pruning
    Args:
        model: torch nn.Module
            Network to prune
        data: namedtuple or dataloader
            Named tuple containing train, validation and test dataloaders
            Alternatively, data can take a dataloader for the train set while
            validation and test dataloaders can be provided under kwargs
        iterations: int
            Number of iterations to train
        rounds: int
            Number of pruning rounds
        prune_net: function
            Function to prune model
        random: bool, default False
            Reset network parameters randomly each pruning round.
            If `False` parameters will be assigned to their original weights
        rewind: float or int, default 0
            Percentage (float) or number (int) of iterations to train before using weights
            as reference in rewinding step in subsequent rounds. When set to 0, original weights are used.
            When `random` is set to `True`, this argument is inoperative.
        recover: str, default None
            Path to folder containing progress from interrupted run.
            Make sure to place the same configurations as the ones used in the previous attempt.
            Additionally, by a functional limitation, the system will repeat the last successful round,
            but this will be fixed in future versions.
    """
    if isinstance(data, DataLoader):
        data = data.datawrapper(
            train=data, validation=kwargs.get("validation"), test=kwargs.get("test")
        )

    
    random = kwargs.get("random", False)
    save = kwargs.get("recover", False)
    recover = bool(save)

    if not recover:
        save = utils._get_save_dir(
            model._get_name(), str(data.train.dataset.dataset), random
        )

    logger.debug(f"save path: {save}")

    # get device for computation
    device = utils._get_device()
    if device:
        model = model.to(device)

    # get number of steps for evaluation (validation or test set)
    step = utils._get_eval_step(data.train)
    original_model = deepcopy(model)
    original_weights = deepcopy(original_model.state_dict())

    def get_weights(random):

        if random:
            original_model._initialize_weights()
            return deepcopy(original_model.state_dict())

        return original_weights

    rewind = kwargs.get("rewind", 0)
    if rewind > 0 and rewind < 1:
        rewind = int(rewind * iterations)

    # recover does not yet work with rewind
    if recover:

        meta = utils._get_meta_file(save)
        init_round, masks = utils.recover_training(save)
        model = prune_net(model)
        placeholder = deepcopy(model.state_dict())
        keys = [label for label in placeholder.keys() if label.endswith("_mask")]
        for k in masks.keys():
            placeholder[[x for x in keys if x.startswith(k)][0]] = masks[k]

        model.load_state_dict(placeholder)
        with torch.no_grad():
            model(next(iter(data.train))[0][0])  # update weights

    else:

        init_round = 0
        # get meta vars of pruning function
        prune_meta = utils._get_prune_meta(prune_net)

        # build meta file
        meta = utils.build_meta(
            model,
            data,
            iterations=iterations,
            rounds=rounds,
            step=step,
            random=random,
            rewind=rewind,
            **prune_meta,
        )

        with open(os.path.join(save, "meta.json"), "w") as f:
            json.dump(meta, f)

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
            correct += (predictions == labels).float().sum().item()

        accuracy = correct / (len(dataloader) * dataloader.batch_size)
        loss = dataloss / len(dataloader)

        return loss, accuracy

    # template for csv file
    template = namedtuple(
        "checkpoint",
        "iteration, train_loss, sparsity, valid_loss, valid_acc, test_loss, test_acc",
    )

    sparsity = prune.sparsity(model) if init_round else 0.0
    cp = template(
        iteration=list(),
        train_loss=list(),
        sparsity=sparsity,
        valid_loss=list(),
        valid_acc=list(),
        test_loss=list(),
        test_acc=list(),
    )  # checkpoint

    # start train-prune rounds
    for r in range(init_round, rounds + 1):

        # create placeholder to avoid any conflicts with the masks created by pruning
        # once pruned `layer.weight` becomes a multiplication of
        # `layer.original_weights` and `layer.weight_mask`
        placeholder = deepcopy(model.state_dict())
        original_weights = get_weights(random)
        keys = [label for label in placeholder.keys() if not label.endswith("_mask")]
        for k in original_weights.keys():
            placeholder[[x for x in keys if x.startswith(k)][0]] = original_weights[k]

        # replace weights
        model.load_state_dict(placeholder)

        i = 0
        min_loss = 1e10
        best_model = dict()
        start = time.time()

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

                # evaluation step
                if (i + 1) % step == 0:

                    if data.test:

                        test_loss, test_acc = evaluate(data.test)
                        cp.test_loss.append(test_loss)
                        cp.test_acc.append(test_acc)

                    if data.validation:

                        valid_loss, valid_acc = evaluate(data.validation)
                        cp.valid_loss.append(valid_loss)
                        cp.valid_acc.append(valid_acc)

                        # 'early stopping' is emulated in this stage according to Frankle and Carbin (2019)
                        # here we store the best performing model according to the validation set
                        if valid_loss < min_loss:
                            best_iter = i
                            min_loss = valid_loss
                            best_model = deepcopy(dict(model.named_parameters()))
                            best_model.update(deepcopy(dict(model.named_buffers())))
                            best_test_loss = test_loss if data.test else None

                else:

                    # adding `nan` to keep columns at the same dimension
                    # if `step == 1` this part is unnecessary
                    if data.test:
                        cp.test_loss.append(float("nan"))
                        cp.test_acc.append(float("nan"))

                    if data.validation:
                        cp.valid_loss.append(float("nan"))
                        cp.valid_acc.append(float("nan"))

                # report each training iteration
                if (i + 1) % len(data.train) == 0:

                    epoch = (i + 1) // len(data.train)
                    loss = sum(cp.train_loss) / len(data.train)
                    duration = time.time() - start
                    start = time.time()

                    message = f"[round: {r} | epoch: {epoch}] "
                    message += f"train: {loss:.3f} "
                    if cp.valid_loss:
                        message += f"validation: {cp.valid_loss[-1]:.4f} "
                    message += f"| sparsity: {cp.sparsity:.0f}% "
                    message += f"| duration: {duration:.1f}s"

                    logger.info(message)

                    # write collected data so far and reset buffer for the next epoch
                    if save:
                        # make cp.sparsity a list
                        cp = cp._replace(sparsity=[cp.sparsity for _ in cp.train_loss])
                        utils.write_epoch(cp, os.path.join(save, str(r)))

                    cp = template(
                        iteration=list(),
                        train_loss=list(),
                        sparsity=sparsity,
                        valid_loss=list(),
                        valid_acc=list(),
                        test_loss=list(),
                        test_acc=list(),
                    )  # checkpoint

                if not r and not random and i == rewind:
                    if i:
                        original_weights = deepcopy(model.state_dict())

                i += 1

        # add summarized round info
        if data.validation:
            meta["round_best_iteration"] = meta["round_best_iteration"][:r] + [
                best_iter
            ]

            if data.test:
                meta["round_test_error"] = meta["round_test_error"][:r] + [
                    best_test_loss
                ]

            if save:
                torch.save(
                    best_model, os.path.join(save, f"{r}/weights_best_model.pth")
                )
                with open(os.path.join(save, "meta.json"), "w") as f:
                    json.dump(meta, f)

        if r < rounds:
            model = prune_net(model)
            sparsity = prune.sparsity(model)
            cp = template(
                iteration=list(),
                train_loss=list(),
                sparsity=sparsity,
                valid_loss=list(),
                valid_acc=list(),
                test_loss=list(),
                test_acc=list(),
            )  # checkpoint

    logger.success("Finished pruning")

    return model
