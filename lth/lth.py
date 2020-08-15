import json
import os
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
    earlystopping: int = 0,
    save: bool = False,
):
    for r in range(rounds):

        prev_loss = 1e10
        streak = 0
        losses = {"train": list(), "test": list()}

        for epoch in range(1, epochs + 1):

            train_loss = 0.0
            test_loss = 0.0

            for inputs, labels in trainloader:

                model.optimizer.zero_grad()

                outputs = model(inputs)
                loss = model.criterion(outputs, labels)
                loss.backward()
                model.optimizer.step()

                train_loss += loss.item()

            train_loss /= trainloader.batch_size
            losses['train'].append(train_loss)

            for inputs, labels in testloader:

                outputs = model(inputs)
                loss = model.criterion(outputs, labels)
                test_loss += loss.item()

            test_loss /= testloader.batch_size
            losses['test'].append(test_loss)

            print(f'[{epoch}] loss: {train_loss:.3f}') if verbose else None

            if earlystopping:
                if abs(train_loss - prev_loss) < tol:
                    streak += 0
                else:
                    streak = 0

                if streak >= earlystopping:
                    print("Early stopping")
                    break

            train_loss = 0.0
            test_loss = 0.0

        if save:
            directory = os.path.join(save, r)
            write_data(model, losses, directory)

    print("Finished Training")

def write_data(model, losses, directory):
    os.makedirs(directory, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(directory, "weights"))
    with open(os.path.join(directory, 'loss.json'), 'w') as f:
        json.dump(losses, f)


