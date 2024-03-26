from torch import nn
from .modify_model_for_verification import modify_model

def mnist_4_1024():
    return nn.Sequential(
        nn.Flatten(),
        nn.ConstantPad1d((0, 0, 0, 0), value=-1),
        nn.Linear(784, 1024),
        nn.ReLU(inplace=True),
        nn.Linear(1024, 1024),
        nn.ReLU(inplace=True),
        nn.Linear(1024, 1024),
        nn.ReLU(inplace=True),
        nn.Linear(1024, 10)
    )

def mnist_4_1024_mod():
    model = nn.Sequential(
        nn.Flatten(),
        nn.ConstantPad1d((0, 0, 0, 0), value=-1),
        nn.Linear(784, 1024),
        nn.ReLU(inplace=True),
        nn.Linear(1024, 1024),
        nn.ReLU(inplace=True),
        nn.Linear(1024, 1024),
        nn.ReLU(inplace=True),
        nn.Linear(1024, 10)
    )

    return modify_model(model)
