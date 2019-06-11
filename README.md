# Neural Optimizer Search
Neural optimizer search based on https://arxiv.org/pdf/1709.07417.pdf

The aim of this experiment is to train a reinforcement learning agent to
search for program _instructions_ that would generate a good neural optimizer.
The instructions are run on a stack machine initialized with a stack with a single element 0.
The instructions can be used to represent a limited set of mathematical equations.
The program halts when the first element 0 is popped from the stack.

For example, SGD with learning rate `0.01` can be represented as:
```
PUSH 1    # Memory slot 1 is the gradient (g)
PUSH_CONST 100
DIV       # Apply learning rate
SUB
POP
```
Since the stack is initialized with 0, this results in the equation `-grad * 0.01`. The final term popped from the stack is used in the weight update rule, e.g, `w += -grad * 0.01`.

The stack machine has several memory slots that can be used for storing additional terms such as momentum. For example, SGD with momentum (with 0.01 learning rate) can be represented as:
```
# Compute momentum
PUSH 3    # Memory slot 3 is empty and initialized with zero
PUSH_CONST 0.9
MUL
PUSH 1    # Memory slot 1 is the gradient (g)
ADD
STORE 3   # Set slot 3 to the computed value (rexecuted each iteration)

PUSH_CONST 100
DIV
SUB
POP
```

See `meta_env/stack_machine.py` for all instructions.

## Running
Python 3.6+ must be installed. To install all project dependencies:

Install dependencies on Ubuntu:
```
apt-get install -y libfontconfig1 libxrender1 libsm6 libxext6 cmake libopenmpi-dev python3-dev zlib1g-dev libglib2.0-0
```

```
pip install -r requirements.txt
```

To train a new model:
```
python -m nos.train
```

Training is performed using PPO with an LSTM policy.