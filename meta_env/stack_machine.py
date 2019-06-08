import torch
import torch.nn as nn
import torch.nn.functional as F

class StackOp():
    def __init__(self, op=None):
        self.arg = op
    def __call__(self, machine):
        self.arg(machine)
    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, self.arg)

class PushOp(StackOp):
    def __call__(self, machine):
        assert self.arg >= 0 and self.arg < len(machine.memory), self.arg
        machine.stack.append(machine.memory[self.arg])

class PushConst(StackOp):
    def __call__(self, machine):
        machine.stack.append(torch.tensor(self.arg, dtype=torch.float))

class PopOp(StackOp):
    def __call__(self, machine):
        machine.stack.pop()

class StoreOp(StackOp):
    def __call__(self, machine):
        machine.memory[self.arg] = machine.stack[-1]
        machine.tainted_memory[self.arg] = True

class UnaryOp(StackOp):
    def __call__(self, machine):
        assert len(machine.stack) > 0, (self.arg, machine.stack)
        machine.stack[-1] = self.arg(machine.stack[-1])

class BinaryOp(StackOp):
    def __call__(self, machine):
        b = machine.stack.pop()
        a = machine.stack.pop()
        if a.is_cuda or b.is_cuda:
            a = a.cuda()
            b = b.cuda()
        machine.stack.append(self.arg(a, b))

class StackMachine():
    def __init__(self):
        self.stack = []
        self.memory = []

    def init(self):
        self.memory = [None for _ in range(mem_size)]
        # A bitmap of whether memory was touched or not
        self.tainted_memory = [False for _ in range(mem_size)]
    
    def reset(self, w, g):
        self.stack = [torch.zeros_like(g)]
        # Fixed memory slots
        self.memory[0] = w
        self.memory[1] = g
        self.tainted_memory[0] = True
        self.tainted_memory[1] = True
        self.history = []

    def __call__(self, opcode):
        op = all_ops[opcode]
        assert len(self.stack) > 0, ('Execute on empty stack', op)
        assert self.is_legal(opcode)

        head = self.stack[-1]
        try:
            # Executes the instruction
            op(self)
            self.history.append(op)
        except Exception as e:
            print('Error executing instruction.', op, self.stack)
            raise e

        if len(self.stack) == 0:
            assert isinstance(op, PopOp)
            # Program ended
            return head

        return None

    def is_legal(self, opcode):
        op = all_ops[opcode]

        if isinstance(op, BinaryOp) and len(self.stack) < 2:
            return False

        if isinstance(op, PushOp) and not self.tainted_memory[op.arg]:
            # Cannot load untained memory
            return False

        if isinstance(op, PopOp) and not (len(self.stack) == 1 or (len(self.history) > 0 and isinstance(self.history[-1], StoreOp))):
            # Can only pop after a store operation or when stack size = 1
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

# The amount of variable to store in memory
mem_size = 8

# All the operators
all_ops = (
    [PopOp()] +
    [PushOp(i) for i in range(mem_size)] +
    [PushConst(-1), PushConst(0), PushConst(1), PushConst(2), PushConst(10), PushConst(100), PushConst(0.9), PushConst(0.99)] +
    # [StoreOp(i) for i in range(mem_size)] +
    [StoreOp(i) for i in range(2, mem_size)] +
    [
        UnaryOp(NamedF('sign', lambda x: torch.sign(x))),
        UnaryOp(NamedF('abs', lambda x: torch.abs(x))),
        UnaryOp(NamedF('cos', lambda x: torch.cos(x))),
        UnaryOp(NamedF('sqrt', lambda x: torch.sqrt(x))),
        UnaryOp(NamedF('log', lambda x: torch.log(x))),
    ] +
    [
        BinaryOp(NamedF('add', lambda a, b: a + b)),
        BinaryOp(NamedF('sub', lambda a, b: a - b)),
        BinaryOp(NamedF('mul', lambda a, b: a * b)),
        BinaryOp(NamedF('div', lambda a, b: a / (b + 1e-8))),
        BinaryOp(NamedF('exp', lambda a, b: a ** b)),
        BinaryOp(NamedF('max', lambda a, b: torch.max(a, b))),
        BinaryOp(NamedF('min', lambda a, b: torch.min(a, b))),
    ]
)