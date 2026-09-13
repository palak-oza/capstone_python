

import torch.nn as nn


def cnn_arc_sim():
    return nn.Sequential(

        # Input: 3 x 32 x 32

        nn.Conv2d(
            in_channels=3,
            out_channels=16,
            kernel_size=3,
            padding=1
        ),
        nn.ReLU(),

        nn.MaxPool2d(kernel_size=2),

        # 32 x 32 -> 16 x 16


        nn.Conv2d(
            in_channels=16,
            out_channels=32,
            kernel_size=3,
            padding=1
        ),
        nn.ReLU(),

        nn.MaxPool2d(kernel_size=2),

        # 16 x 16 -> 8 x 8


        nn.Flatten(),

        nn.Linear(32 * 8 * 8, 64),
        nn.ReLU(),

        nn.Linear(64, 10)
    )