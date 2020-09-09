import os
import sys
import json
from functools import partial

import torch
from loguru import logger

import data
import models
import lth
import prune
from _cli import parser


args = parser.parse_args()

# env var settings
if args.gpu:
    lth.utils.set_device("cuda")
if args.quiet:
    lth.utils.set_verbosity("WARNING")
if args.seed:
    lth.utils.set_seed(args.seed)
if args.step:
    lth.utils.set_eval_step(args.step)

# get first match in _dispatcher
modelkey = [k for k in models._dispatcher.keys() if args.net.lower() in k][0]
load_model = models._dispatcher[modelkey]
# params is used to discard all `None` args from call
# this way we assure that the assigned default values for
# each model will called when this happens
params = {"optim": args.optim, "lr": args.learn_rate}
params = {key: value for key, value in params.items() if value is not None}
model = load_model(**params)

# get first match in _dispatcher
datakey = [k for k in data._dispatcher.keys() if args.dataset.lower() in k][0]
load_dataset = data._dispatcher[datakey]

params = {"root": os.path.join(args.data, datakey), "train_batch_size": args.batch_size}
params = {key: value for key, value in params.items() if value is not None}
dataloader = load_dataset(**params)

# prune method selection
if args.prune_rate and args.prune_global:
    prune_method = partial(prune.prune_conv_globally, rate=args.prune_rate)

elif args.prune_rate and args.fc_rate:

    def prune_method(model):

        model = prune.prune_fc(model, args.fc_rate)
        model = prune.prune_conv(model, args.prune_rate)

        return model


elif args.fc_rate:
    prune_method = partial(prune.prune_fc, rate=args.prune_fc)

else:
    prune_method = partial(prune.prune_all, rate=args.prune_rate)

params = {"random": args.random, "rewind": args.rewind, "recover": args.recover}
params = {key: value for key, value in params.items() if value}

lth.iterative_pruning(model, dataloader, args.iter, args.rounds, prune_method, **params)
