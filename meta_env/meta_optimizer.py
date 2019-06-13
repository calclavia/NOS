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
                if 'm_buffer' not in state:
                    m = state['m_buffer'] = torch.clone(grad).detach()
                else:
                    m = state['m_buffer']
                    beta1 = 0.9
                    m.mul_(beta1).add_(1 - beta1, grad)

                if 'v_buffer' not in state:
                    v = state['v_buffer'] = torch.zeros_like(p.data)
                else:
                    v = state['v_buffer']
                    beta2 = 0.999
                    v.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                
                # Grab executor
                if 'executor' not in state:
                    state['executor'] = group['machine']
                    state['executor'].init()
                
                # Reset stack and override memory slots
                executor = state['executor']
                executor.reset(p.data, grad, m, v)

                for instr in instrs:
                    # Execute instruction
                    res = executor(instr)
                    
                    if res is not None:
                        # Finish execution
                        p.data.add_(res)
                        break
        return loss