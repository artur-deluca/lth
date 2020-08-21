import json
import lth
import torch
from datetime import datetime
from os import path, makedirs
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from lth.models.utils import lr, optim


parser = ArgumentParser(
    description="Run experiments with Iterative Prunning, identifying Lottery Tickets",
    formatter_class=ArgumentDefaultsHelpFormatter
)
parser.add_argument(
    "net",
    type=str,
    help="Network architecture to use. For more info run `utils.models.models`",
)
parser.add_argument("dataset", type=str, help="Dataset type (MNIST or CIFAR10)")
parser.add_argument(
    "-p", "--data", metavar='', default="./datasets/", type=str, help="Path to root dataset folder"
)
parser.add_argument("-bs", "--batch_size", metavar='', default=60, type=int, help="Dataloader's batch size")
parser.add_argument("-o", "--optim", metavar='', default=optim, type=str, help="Model's optimizer")
parser.add_argument(
    "-lr", "--learn_rate", metavar='', default=lr, type=float, help="Learning rate"
)
parser.add_argument("-i", "--iter", metavar='', default=int(60e3), type=int, help="Training epochs")
parser.add_argument("-r", "--rounds", metavar='', default=15, type=int, help="Prunning rounds")
parser.add_argument(
    "-pr", "--prune_rate", metavar='', default=0.2, type=float, help="Prunning rate 0-.99"
)
parser.add_argument("-s", "--save", metavar='', default="./experiments/", type=str, help="Directory to store the experiments")
parser.add_argument("-rs", "--seed", metavar='', default=None, type=int, help="Random seed")
parser.add_argument(
    "-fc",
    "--fc_rate",
    metavar='',
    default=None,
    type=int,
    help="Different prunning rate for Fully Connected layers",
)
parser.add_argument(
    "--batch_norm", action="store_true", help="Use batch norm in architecture"
)
parser.add_argument(
    "--prune_global", action="store_true", help="Global prunnning instead of layer-wise"
)
parser.add_argument(
    "--gpu", action="store_true", help="Allow for GPU usage"
)
parser.add_argument("--verbose", action="store_true", help="Verbosity mode")

parser.set_defaults(batch_norm=False)
parser.set_defaults(prune_global=False)
parser.set_defaults(verbose=False)
parser.set_defaults(gpu=False)

args = parser.parse_args()
args.dataset = args.dataset.lower()

directory = str()
args.gpu = torch.device('cuda') if args.gpu else args.gpu

if __name__ == "__main__":

    if str(args.seed).isnumeric():
        torch.manual_seed(args.seed)


    model = lth.models._dispatcher[args.net](
        optim=args.optim, lr=args.learn_rate, batch_norm=args.batch_norm
    )
    datakey = [k for k in lth.data._dispatcher.keys() if args.dataset in k][0]
    train, validation, test = lth.data._dispatcher[datakey](path.join(args.data, datakey), batch_size=args.batch_size)

    if args.save:
        net = args.net + f'{"_bn_" if args.batch_norm else ""}'
        f = f'{datetime.now().strftime("%m_%d_%Y_%I_%M")}_{net}_{args.dataset}'
        directory = path.join(args.save, f)
        makedirs(directory, exist_ok=True)

        meta = lth.build_meta(
            model,
            train,
            net=net,
            epochs=args.epochs,
            prune_rate=args.prune_rate,
            rounds=args.rounds,
            globally=args.prune_global,
            fc_rate=args.fc_rate,
            gpu=args.gpu,
            seed=str(args.seed)
        )

        with open(path.join(directory, "meta.json"), "w") as f:
            json.dump(meta, f)

    lth.iterative_pruning(
        model,
        train,
        validation,
        test,
        args.epochs,
        args.rounds,
        args.prune_rate,
        args.verbose,
        directory,
        args.prune_global,
        args.fc_rate,
        device=args.gpu
    )
