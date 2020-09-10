# Testing the Lottery Ticket Hypothesis
[![report](https://img.shields.io/badge/Report-pdf-lightgrey)](https://nbviewer.jupyter.org/github/artur-deluca/lth/blob/master/Report.pdf)

This work intends to replicate some of the experiments of [Frankle and Carbin's Lottery Ticket Hypothesis](https://openreview.net/forum?id=rJl-b3RcF7).

> The authors also have a great framework for these experiments. Check out [OpenLTH](https://github.com/facebookresearch/open_lth/)!

To execute any of the original experiments run:
```bash
# don't forget to install requirements first
> python lth --help
usage: lth [-h] [-p] [--batch_size] [-o] [-lr] [-i] [-r] [-es] [-rw] [-pr]
           [--recover] [-s] [-rs] [-fc] [--prune_global] [--gpu] [--quiet]
           [--random]
           net dataset

Run experiments with Iterative Prunning, identifying Lottery Tickets

positional arguments:
  net                  Network architecture to use. For more info run
                       `utils.models.models`
  dataset              Dataset type (MNIST or CIFAR10)

optional arguments:
  -h, --help           show this help message and exit
  -p , --data          Path to root dataset folder (default: ./datasets/)
  --batch_size         Dataloader's batch size (training) (default: None)
  -o , --optim         Model's optimizer (default: None)
  -lr , --learn_rate   Learning rate (default: None)
  -i , --iter          Training iterations (default: 50000)
  -r , --rounds        Prunning rounds (default: 26)
  -es , --step         Evaluate validation and test set every x steps. To
                       evaluate every epoch, use -1 (default: None)
  -rw , --rewind       Number of iterations to train in the first round before
                       using weights as reference for later rounds. Set rewind
                       between (0, 1) to use it as a percetage. (default: 0)
  -pr , --prune_rate   Prunning rate 0-.99 (default: 0.2)
  --recover            Recover/resume interrupted training procedure (default:
                       None)
  -s , --save          Directory to store the experiments (default:
                       ./experiments/)
  -rs , --seed         Random seed (default: None)
  -fc , --fc_rate      Different prunning rate for Fully Connected layers
                       (default: None)
  --prune_global       Global prunnning instead of layer-wise (default: False)
  --gpu                Allow for GPU usage (default: False)
  --quiet              Verbosity mode (default: False)
  --random             Random initialization (default: False)

# then, for instance
> python lth lenet mnist --rounds 20 --prune_rate 0.2
```
> The code was written in Python 3.7 using Pytorch's [pruning module](https://pytorch.org/tutorials/intermediate/pruning_tutorial.html)
You can also create a model and make your own pruning experiments:

```python
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

import lth

# if data is not there, it will prompt you to download it
# containing train, validation and test dataloaders
dataloader = lth.data.load_MNIST('./datasets/mnist')

class Custom_Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.net = nn.Sequential(
                nn.Linear(784, 200),
                nn.Linear(200, 100),
                nn.Linear(100, 10)
        )
        # there are some required attributes
        self.optim = torch.optim.Adam(self.parameters(), lr=0.05)
        self.optim.name = 'adam'
        self._initialize_weights()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.net(x)
    
    # defining required initialization method
    def _initialize_weights(self):
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 0.01)
                    nn.init.constant_(m.bias, 0)

# make your own pruning method   
def prune(net):
    layers = lth.prune.fetch_layers(net) # fetch all parameters to be pruned
    for (layer, param_type) in layers: 
        prune.l1_unstructured(layer, name=param_type, amount=0.11)
    return net

iterations = 1200
rounds = 15

lth.iterative_pruning(
    net, dataloader, iterations, rounds, prune
)


```

