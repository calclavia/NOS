import torch
import torch.nn as nn
import torch.nn.functional as F

class NamedF():
    def __init__(self, name, f):
        self.name = name
        self.f = f
    def __call__(self, *args, **kwargs):
        return self.f(*args, **kwargs)
    def __str__(self):
        return self.name
    def __repr__(self):
        return self.name

class Instr():
    """ An instruction to be executed. """
    def __init__(self, op=None):
        self.arg = op
    def __call__(self, machine):
        self.arg(machine)
    def can_execute(self, machine):
        """ Returns True if this instruction can be exceuted """
        return True
    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, self.arg)

class PushOp(Instr):
    """ Pushes memory onto the stack """
    def __call__(self, machine):
        assert self.arg >= 0 and self.arg < len(machine.memory), self.arg
        machine.stack.append(machine.memory[self.arg])
        
    def can_execute(self, machine):
        # Cannot load untained memory
        return machine.tainted_memory[self.arg]

class PushConst(Instr):
    """ Pushes constant onto the stack """
    def __call__(self, machine):
        machine.stack.append(torch.tensor(self.arg, dtype=torch.float))

class PopOp(Instr):
    """ Pops memory from the stack """
    def __call__(self, machine):
        machine.stack.pop()
        
    def can_execute(self, machine):
        # Can only pop after a store operation or when stack size = 1
        return len(machine.stack) == 1 or (len(machine.history) > 0 and isinstance(machine.history[-1], StoreOp))

class StoreOp(Instr):
    """ Stores memory onto the stack """
    def __call__(self, machine):
        machine.memory[self.arg] = machine.stack[-1]
        machine.tainted_memory[self.arg] = True

class UnaryOp(Instr):
    """ Applies a unary operation to the top of the stack """
    def __call__(self, machine):
        assert len(machine.stack) > 0, (self.arg, machine.stack)
        machine.stack[-1] = self.arg(machine.stack[-1])

class BinaryOp(Instr):
    """ Applies a binary operation to the top 2 items on the stack """
    def __call__(self, machine):
        b = machine.stack.pop()
        a = machine.stack.pop()
        if a.is_cuda:
            b = b.to(device=a.device)
        if b.is_cuda:
            a = a.to(device=b.device)
        machine.stack.append(self.arg(a, b))

    def can_execute(self, machine):
        return len(machine.stack) >= 2