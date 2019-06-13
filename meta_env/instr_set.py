import torch
import torch.nn as nn
import torch.nn.functional as F
from .instr import *

# The amount of variable to store in memory
mem_size = 8

const_ops = [PushConst(1), PushConst(2), PushConst(10), PushConst(100)]

unary_ops = [
    UnaryOp(NamedF('cos', lambda x: torch.cos(x))),
    UnaryOp(NamedF('sign', lambda x: torch.sign(x))),
    UnaryOp(NamedF('sqrtabs', lambda x: x.abs().sqrt())),
    UnaryOp(NamedF('logabs', lambda x: x.abs().log())),
]

binary_ops = [
    BinaryOp(NamedF('add', lambda a, b: a + b)),
    BinaryOp(NamedF('sub', lambda a, b: a - b)),
    BinaryOp(NamedF('mul', lambda a, b: a * b)),
    BinaryOp(NamedF('div', lambda a, b: a / (b + 1e-8))),
    BinaryOp(NamedF('exp', lambda a, b: a ** b)),
    BinaryOp(NamedF('max', lambda a, b: torch.max(a, b))),
    BinaryOp(NamedF('min', lambda a, b: torch.min(a, b))),
]

# All the operators
stack_instr_set = (
    [PopOp()] +
    [PushOp(i) for i in range(mem_size)] +
    # [StoreOp(i) for i in range(2, mem_size)] +
    const_ops +
    unary_ops + binary_ops
)

tree_instr_set = (
    [PushOp(i) for i in range(mem_size)] +
    const_ops +
    [UnaryOp(NamedF('identity', lambda x: x))] +
    unary_ops + binary_ops
)    
