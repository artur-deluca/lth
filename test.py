from functools import partial

import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

import lth

SEED = 423

class Moons(Dataset):

    def __init__(self, X, y):
            self.X = torch.from_numpy(X.astype(np.float32))
            self.y = torch.from_numpy(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

    def dataset(self):
        return "Moons"


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.net = nn.Sequential(
                nn.Linear(2, 100),
                nn.Linear(100, 100),
                nn.Linear(100, 2)
        )
        self.optim = torch.optim.Adam(self.parameters(), lr=0.05)
        self.optim.name = 'adam'
        self._initialize_weights()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.net(x)

    def _initialize_weights(self):
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 0.01)
                    nn.init.constant_(m.bias, 0)

if __name__ == '__main__':
    X, y = make_moons(n_samples=1000, shuffle=True, noise=0.3, random_state=SEED) 

    # create train and test indices
    train, test = train_test_split(list(range(X.shape[0])), test_size=.2, random_state=SEED)
    train, valid = train_test_split(train, test_size=.1, random_state=SEED)



    dataset = Moons(X, y)
    train = DataLoader(dataset, batch_size=50, sampler=SubsetRandomSampler(train))
    validation = DataLoader(dataset, batch_size=50, sampler=SubsetRandomSampler(valid))
    test = DataLoader(dataset, batch_size=50, sampler=SubsetRandomSampler(test))



    model = Model()
    dataloader = lth.data.datawrapper(train=train, validation=validation, test=test)

    lth.utils.set_seed(SEED)
    lth.utils.set_eval_step(-1)
    prune_method = partial(lth.prune.prune_all, rate=0.15)

    lth.iterative_pruning(
        model, dataloader, 1200, 15, prune_method 
    )
