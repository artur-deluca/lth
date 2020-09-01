import os
import sys
import json
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import torch
from loguru import logger

import data
import models
import lth
import prune

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
parser.add_argument("-o", "--optim", metavar='', default=None, type=str, help="Model's optimizer")
parser.add_argument(
    "-lr", "--learn_rate", metavar='', default=None, type=float, help="Learning rate"
)
parser.add_argument("-i", "--iter", metavar='', default=int(60e3), type=int, help="Training epochs")
parser.add_argument("-r", "--rounds", metavar='', default=26, type=int, help="Prunning rounds")
parser.add_argument("-es", "--step", metavar='', default=None, type=int, help="Evaluate validation and test set every x steps. For every epoch use -1")
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
    type=float,
    help="Different prunning rate for Fully Connected layers",
)
parser.add_argument(
    "--prune_global", action="store_true", help="Global prunnning instead of layer-wise"
)
parser.add_argument(
    "--gpu", action="store_true", help="Allow for GPU usage"
)

parser.add_argument("--quiet", action="store_true", help="Verbosity mode")
parser.add_argument("--random", action="store_true", help="Random initialization")

parser.set_defaults(prune_global=False)
parser.set_defaults(quiet=False)
parser.set_defaults(random=False)
parser.set_defaults(gpu=False)

args = parser.parse_args()

if args.gpu: lth.utils.set_device('cuda')
if args.quiet: lth.utils.set_verbosity('WARNING')
if args.seed: lth.utils.set_seed(args.seed)
if args.step: lth.utils.set_eval_step(args.step)

params = {'optim': args.optim, 'lr': args.learn_rate}
params = {key: value for key, value in params.items() if value is not None}
model = models._dispatcher[args.net](**params)

datakey = [k for k in data._dispatcher.keys() if args.dataset.lower() in k][0]
dataloader = data._dispatcher[datakey](os.path.join(args.data, datakey), batch_size=args.batch_size)

prune_container = prune.prune_container(rate=args.prune_rate, fc_rate=args.fc_rate, globally=args.prune_global)

lth.iterative_pruning(
    model,
    dataloader,
    args.iter,
    args.rounds,
    prune_container,
    args.random
)
