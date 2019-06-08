import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
from torch.distributions import Categorical
from model import ConvModel
from . import meta_optimizer
from .meta_optimizer import all_ops, StackMachine, MetaOptimizer
from .data import create_datasets
from gym import Env, logger
from gym import spaces
import numpy as np

class MetaEnv(Env):
    def __init__(self, dataset='cifar10', batch_size=128, cuda=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.cuda = cuda
        self.action_space = spaces.Discrete(len(all_ops))
        self.observation_space = spaces.Discrete(len(all_ops))
        self.train_loader, self.val_loader = create_datasets(dataset, batch_size, cuda)

    def reset(self):
        self.executor = StackMachine()
        # Initialize dummy variable
        self.executor.init(torch.tensor(1.0))
        self.instrs = []
        return 0

    def get_valid_actions(self):
        """ Get a vector of valid actions """
        valid_actions = [self.executor.is_legal(action) for action in range(len(all_ops))]
        assert True in valid_actions
        return valid_actions

    def step(self, action):
        assert self.action_space.contains(action)
        
        done = False
        state = action
        reward = 0
        self.instrs.append(action)

        ops = [all_ops[x] for x in self.instrs]

        if self.executor(action) is not None:
            done = True
            train_failed = False      
            model = ConvModel()

            if self.cuda:
                model.cuda()

            optimizer = MetaOptimizer(model.parameters(), self.instrs)

            # print('Training meta model')
            num_epochs = 1
            # total_iters = num_epochs * len(self.train_loader)
            for e in range(num_epochs):
                model.train()
                # Train on training set
                iterations = 0
                for x, y in self.train_loader:
                    if self.cuda:
                        x, y = x.cuda(), y.cuda()

                    y_pred = model(x)
                    loss = F.cross_entropy(y_pred, y)
                    model.zero_grad()

                    if torch.isnan(loss).any().item():
                        train_failed = True
                        print('Failed train', iterations, loss.item(), ops)
                        break
                    
                    iterations += 1
                    loss.backward()
                    optimizer.step()

                if train_failed:
                    break

            if not train_failed:
                # Validation set
                with torch.no_grad():
                    model.eval()
                    accs = []
                    for x, y in self.val_loader:
                        if self.cuda:
                            x, y = x.cuda(), y.cuda()
                        y_pred = model(x)
                        accs += (y_pred.argmax(dim=-1) == y).tolist()
                reward += np.mean(accs)
                print('Validation Acc:', np.mean(accs), ops)
        return state, reward, done, {}
