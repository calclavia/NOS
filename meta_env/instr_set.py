import torch
import torch.nn as nn
import torch.nn.functional as F
from .instr import *

# The amount of variable to store in memory
mem_size = 8

# All the operators
stack_instr_set = (
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
        UnaryOp(NamedF('logabs', lambda x: torch.log(torch.abs(x)))),
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

tree_instr_set = (
    [PushOp(i) for i in range(mem_size)] +
    [PushConst(-1), PushConst(0), PushConst(1), PushConst(2), PushConst(10), PushConst(100), PushConst(0.9), PushConst(0.99)] +
    [
        UnaryOp(NamedF('identity', lambda x: x)),
        UnaryOp(NamedF('sign', lambda x: torch.sign(x))),
        UnaryOp(NamedF('sqrtabs', lambda x: x.abs().sqrt())),
        UnaryOp(NamedF('logabs', lambda x: x.abs().log())),
    ] +
    [
        BinaryOp(NamedF('add', lambda a, b: a + b)),
        BinaryOp(NamedF('sub', lambda a, b: a - b)),
        BinaryOp(NamedF('mul', lambda a, b: a * b)),
        BinaryOp(NamedF('div', lambda a, b: a / (b + 1e-8))),
        BinaryOp(NamedF('exp', lambda a, b: a ** b)),
        BinaryOp(NamedF('keepleft', lambda a, b: a))
    ]
)    
