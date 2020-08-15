import json
import lth
import torch
from datetime import datetime
from os import path, makedirs
from argparse import ArgumentParser


parser = ArgumentParser(
    description="Run experiments with Iterative Prunning, identifying Lottery Tickets"
)
parser.add_argument(
    "net",
    type=str,
    help="Network architecture to use. For more info run `utils.models.models`",
)
parser.add_argument("dataset", type=str, help="Dataset type (MNIST or CIFAR10)")
parser.add_argument(
    "--batch_norm", action="store_true", help="Use batch norm in architecture"
)
parser.add_argument(
    "-p", "--data", default="./datasets/", type=str, help="Path to root dataset folder"
)
parser.add_argument("-e", "--epochs", default=100, type=int, help="Training epochs")
parser.add_argument("-r", "--rounds", default=20, type=int, help="Prunning rounds")
parser.add_argument(
    "-pr", "--prune_rate", default=0.2, type=float, help="Prunning rate 0-.99"
)
parser.add_argument("--verbose", action="store_true", help="Verbosity mode")
parser.add_argument(
    "-es", "--earlystopping", default=0, type=int, help="Early stopping epochs"
)
parser.add_argument("-s", "--save", default="./experiments/", type=str, help="")

parser.set_defaults(batch_norm=False)
parser.set_defaults(verbose=False)

args = parser.parse_args()
args.dataset = args.dataset.lower()
directory = ""

if __name__ == "__main__":

    torch.manual_seed(344)

    model = lth.models._dispatcher[args.net](batch_norm=args.batch_norm)
    datakey = [k for k in lth.data._dispatcher.keys() if args.dataset in k][0]
    train, test = lth.data._dispatcher[datakey](path.join(args.data, datakey))


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
        )

        with open(path.join(directory, 'meta.json'), 'w') as f:
            json.dump(meta, f)

    lth.iterative_prunning(
        model,
        train,
        test,
        args.epochs,
        args.rounds,
        args.prune_rate,
        args.verbose,
        directory
    )
