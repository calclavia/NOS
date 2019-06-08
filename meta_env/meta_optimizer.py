import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer
from .stack_machine import StackMachine

class MetaOptimizer(Optimizer):
    def __init__(self, params, instrs):
        defaults = dict(instrs=instrs)
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
                
                # State initialization
                if len(state) == 0:
                    state['executor'] = StackMachine()
                    state['executor'].init()
                
                # Reset stack and override memory slots
                executor = state['executor']
                executor.reset(p.data, grad)

                for instr in instrs:
                    # Execute instruction
                    res = executor(instr)
                    
                    if res is not None:
                        # Finish execution
                        p.data.add_(res)
                        break
        return loss