import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer
from .stack_machine import StackMachine

class MetaOptimizer(Optimizer):
    def __init__(self, params, machine, instrs):
        defaults = dict(machine=machine, instrs=instrs)
        super().__init__(params, defaults)

    def step(self, closure=None):
        """
        Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                
                state = self.state[p]
                instrs = group['instrs']

                # Compute momentum
                if 'momentum_buffer' not in state:
                    m = state['momentum_buffer'] = torch.clone(grad).detach()
                else:
                    m = state['momentum_buffer']
                    m.mul_(0.9).add_(1 - 0, grad)
                
                # Grab executor
                if 'executor' not in state:
                    state['executor'] = group['machine']
                    state['executor'].init()
                
                # Reset stack and override memory slots
                executor = state['executor']
                executor.reset(p.data, grad, m)

                for instr in instrs:
                    # Execute instruction
                    res = executor(instr)
                    
                    if res is not None:
                        # Finish execution
                        p.data.add_(res)
                        break
        return loss