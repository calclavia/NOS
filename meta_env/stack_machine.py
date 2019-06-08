import torch
import torch.nn as nn
import torch.nn.functional as F

# The amount of variable to store in memory
mem_size = 8

# All the operators
all_ops = [PushOp(i) for i in range(mem_size)] +\
[PopOp(i) for i in range(mem_size)] + \
[PushConst(-1), PushConst(0), PushConst(1), PushConst(2), PushConst(10), PushConst(100)] +\
[
    UnaryOp(NamedF('sign', lambda x: torch.sign(x))),
    UnaryOp(NamedF('abs', lambda x: torch.abs(x))),
    UnaryOp(NamedF('sin', lambda x: torch.sin(x))),
    UnaryOp(NamedF('cos', lambda x: torch.cos(x))),
    UnaryOp(NamedF('sqrt', lambda x: torch.sqrt(x))),
    UnaryOp(NamedF('log', lambda x: torch.log(x))),
] + [
    BinaryOp(NamedF('add', lambda a, b: a + b)),
    BinaryOp(NamedF('sub', lambda a, b: a - b)),
    BinaryOp(NamedF('mul', lambda a, b: a * b)),
    BinaryOp(NamedF('div', lambda a, b: a / (b + 1e-8))),
    BinaryOp(NamedF('exp', lambda a, b: a ** b)),
    BinaryOp(NamedF('max', lambda a, b: torch.max(a, b))),
    BinaryOp(NamedF('min', lambda a, b: torch.min(a, b))),
]

class StackOp():
    def __init__(self, op):
        self.op = op
    def __call__(self, stack, memory):
        self.op(stack, memory)
    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, self.op)

class PushOp(StackOp):
    def __call__(self, stack, memory):
        assert self.op >= 0 and self.op < len(memory), self.op
        stack.append(memory[self.op])

class PushConst(StackOp):
    def __call__(self, stack, memory):
        stack.append(torch.tensor(self.op, dtype=torch.float))

class PopOp(StackOp):
    def __call__(self, stack, memory):
        memory[self.op] = stack.pop()

class UnaryOp(StackOp):
    def __call__(self, stack, memory):
        assert len(stack) > 0, (self.op, stack)
        stack[-1] = self.op(stack[-1])

class BinaryOp(StackOp):
    def __call__(self, stack, memory):
        b = stack.pop()
        a = stack.pop()
        if a.is_cuda or b.is_cuda:
            a = a.cuda()
            b = b.cuda()
        stack.append(self.op(a, b))

class StackMachine():
    def __init__(self):
        self.stack = []
        self.memory = []
    def init(self, x):
        self.stack = [x]
        self.memory = [torch.zeros_like(x) for _ in range(mem_size)]
    def __call__(self, opcode):
        op = all_ops[opcode]
        assert len(self.stack) > 0, ('Execute on empty stack', op)
        assert self.is_legal(opcode)

        try:
            # Executes the instruction
            op(self.stack, self.memory)
        except Exception as e:
            print('Error executing instruction.', op, self.stack)
            raise e

        if len(self.stack) == 0:
            assert isinstance(op, PopOp)
            # Program ended
            return self.memory[op.op]
        return None

    def is_legal(self, opcode):
        op = all_ops[opcode]
        if isinstance(op, BinaryOp) and len(self.stack) < 2:
            return False
        return True

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