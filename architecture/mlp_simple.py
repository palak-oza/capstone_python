import torch.nn as nn


def mlp_arc_sim():
    return nn.Sequential(

        nn.Flatten(),

        nn.Linear(3072, 44),
        nn.ReLU(),

        nn.Linear(44, 10)
    )
