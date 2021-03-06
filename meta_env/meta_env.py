import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
from torch.distributions import Categorical
from . import meta_optimizer
from .model import ConvModel

# from .instr_set import stack_instr_set as all_ops
from .instr_set import tree_instr_set as all_ops
from .meta_optimizer import MetaOptimizer
from .stack_machine import StackMachine, TreeMachine

from .data import create_datasets
from gym import Env, logger
from gym import spaces
from copy import deepcopy
import numpy as np

class MetaEnv(Env):
    def __init__(self, dataset='cifar10', batch_size=128, device=torch.device('cpu')):
        self.dataset = dataset
        self.device = device
        self.observation_space = spaces.Box(low=-100, high=100, shape=(len(all_ops) + 1,), dtype=np.float32)
        self.action_space = spaces.Discrete(len(all_ops))
        self.train_loader, self.val_loader = create_datasets(dataset, batch_size)
        self.model = ConvModel()

        self.assess_interval = 20
        self.max_no_improve = 4
        self.max_instrs = 20

    def reset(self):
        self.executor = TreeMachine(all_ops)
        # Initialize dummy variable
        self.executor.reset(torch.tensor(0.0), torch.tensor(0.1), torch.tensor(0.2), torch.tensor(0.3))
        self.instrs = []

        action_features = [0 for _ in range(len(all_ops))]
        state = np.array(action_features + self.get_stack_features())
        return state

    def get_valid_actions(self):
        """ Get a vector of valid actions """
        valid_actions = [self.executor.can_execute(action) for action in range(len(all_ops))]
        assert True in valid_actions
        return valid_actions

    def step(self, action):
        # Test examples that should give baseline results.
        # override = [2, 29, 0]       # Stack instr
        # override = [1, 12, 8, 12, 18] # Tree instr
        # action = override[len(self.instrs)] if len(override) > len(self.instrs) else 0

        assert self.action_space.contains(action)
        
        done = False
        reward = 0
        self.instrs.append(action)
        is_invalid = torch.isnan(self.executor.stack[-1]).any().item() or torch.isinf(self.executor.stack[-1]).any().item()

        ops = [all_ops[x] for x in self.instrs]

        if is_invalid or len(self.instrs) > self.max_instrs:
            # Early terminate if nan
            done = True
            res = 0
            print('Early terminate', ops)
        else:
            res = self.executor(action)
            if res is not None:
                # Train a model
                res = res.item()

                done = True
                train_failed = False      
                model = deepcopy(self.model).to(self.device)

                optimizer = MetaOptimizer(model.parameters(), self.executor, self.instrs)

                best_loss = float('inf')
                train_losses = []
                no_improv = 0

                num_epochs = 3
                iterations = 0
                # total_iters = num_epochs * len(self.train_loader)
                for e in range(num_epochs):
                    model.train()
                    # Train on training set
                    for x, y in self.train_loader:
                        x, y = x.to(self.device), y.to(self.device)
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

                        train_losses.append(loss.item())

                        if iterations % self.assess_interval == 0:
                            avg_loss = np.mean(train_losses[-self.assess_interval:])

                            if avg_loss < best_loss:
                                best_loss = avg_loss
                                no_improv = 0
                            else:
                                no_improv += 1
                        
                        if no_improv > self.max_no_improve:
                            # print('Break training', iterations,
                            #     np.mean(train_losses[-self.assess_interval:]) /  np.mean(train_losses[:self.assess_interval]))
                            break

                    if train_failed:
                        break

                if not train_failed:
                    # Validation set
                    with torch.no_grad():
                        model.eval()
                        accs = []
                        for x, y in self.val_loader:
                            x, y = x.to(self.device), y.to(self.device)
                            y_pred = model(x)
                            accs += (y_pred.argmax(dim=-1) == y).tolist()
                    reward += np.mean(accs)
                    print('Validation Acc:', np.mean(accs), ops)

        # Reward shaping
        # if done:
        #     # Give points if gradient is used (it has to be used somewhere!)
        #     reward += 0.01 if 2 in self.instrs else 0

        # Features
        action_features = [a == action for a in range(len(all_ops))]
        state = np.array(action_features + self.get_stack_features(res))
        return state, reward, done, {}

    def get_stack_features(self, res=0):
        bound = 100
        stack_features = [min(max(self.executor.stack[-1], -bound), bound) if len(self.executor.stack) > 0 else res]
        return stack_features if np.isfinite(stack_features) else [0 for _ in stack_features]