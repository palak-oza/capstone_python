import torch.nn as nn

def mlp_arc():
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(3072, 512),
        nn.ReLU(),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, 10)
    )



