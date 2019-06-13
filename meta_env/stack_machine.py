import torch
import torch.nn as nn
import torch.nn.functional as F
from .instr import *

class StackMachine():
    def __init__(self, instr_set, mem_size=9):
        self.instr_set = instr_set
        self.mem_size = mem_size
        self.init()

    def init(self):
        self.stack = []
        self.history = []
        # Memory buffer
        self.memory = [None for _ in range(self.mem_size)]
        # A bitmap of whether memory was touched or not
        self.tainted_memory = [False for _ in range(self.mem_size)]
    
    def reset(self, w, g, m, v):
        """ Reset before one iteration of execution """
        self.stack = [torch.zeros_like(g)]
        self.history = []
        # Fixed memory slots
        self.memory[0] = w
        self.memory[1] = g
        self.memory[2] = g ** 2
        self.memory[3] = m
        self.memory[4] = v
        self.init_mem_size = 5

        for i in range(self.init_mem_size):
            self.tainted_memory[i] = True

    def __call__(self, opcode):
        op = self.instr_set[opcode]
        assert len(self.stack) > 0, ('Execute on empty stack', op)
        assert self.can_execute(opcode), op

        head = self.execute(op)

        if len(self.stack) == 0:
            # Program ended
            return head

        return None
    
    def execute(self, instr):
        head = self.stack[-1]

        try:
            # Executes the instruction
            instr(self)
            self.history.append(instr)
        except Exception as e:
            print('Error executing instruction.', instr, self.stack)
            raise e
        return head

    def can_execute(self, opcode):
        op = self.instr_set[opcode]
        return op.can_execute(self)

class TreeMachine(StackMachine):
    """ Based on https://arxiv.org/pdf/1709.07417.pdf """
    def reset(self, w, g, m, v):
        super().reset(w, g, m, v)
        self.stage = 0
        self.tree_size = 0

    def execute(self, instr):
        head = super().execute(instr)

        if self.stage == 4:
            if self.tree_size == 3:
                # Finish computing
                super().execute(BinaryOp(NamedF('sub', lambda a, b: a - b)))
                head = self.stack[-1]
                super().execute(PopOp())
            else:
                # Store the computed value
                super().execute(StoreOp(self.tree_size + self.init_mem_size))
                head = self.stack[-1]
                super().execute(PopOp())
            self.tree_size += 1

        self.stage = (self.stage + 1) % 5
        return head

    def can_execute(self, opcode):
        if not super().can_execute(opcode):
            return False
        op = self.instr_set[opcode]

        if self.stage == 0 or self.stage == 2: 
            return isinstance(op, (PushOp, PushConst))
        elif self.stage == 1 or self.stage == 3: 
            return isinstance(op, UnaryOp)
        elif self.stage == 4: 
            return isinstance(op, BinaryOp)
        else:
            raise Exception('Invalid stage')
