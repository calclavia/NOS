import torch
import torch.nn as nn
import torch.nn.functional as F

class StackMachine():
    def __init__(self, instr_set, mem_size=8):
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
    
    def reset(self, w, g):
        """ Reset before one iteration of execution """
        self.stack = [torch.zeros_like(g)]
        self.history = []
        # Fixed memory slots
        self.memory[0] = w
        self.memory[1] = g
        self.tainted_memory[0] = True
        self.tainted_memory[1] = True

    def __call__(self, opcode):
        op = self.instr_set[opcode]
        assert len(self.stack) > 0, ('Execute on empty stack', op)
        assert self.can_execute(opcode), op

        head = self.stack[-1]
        try:
            # Executes the instruction
            op(self)
            self.history.append(op)
        except Exception as e:
            print('Error executing instruction.', op, self.stack)
            raise e

        if len(self.stack) == 0:
            # Program ended
            return head

        return None

    def can_execute(self, opcode):
        op = self.instr_set[opcode]
        return op.can_execute(self)