import torch.nn as nn

def cnn_arc():
    return nn.Sequential(

        # Input: 3 x 32 x 32

        nn.Conv2d(
            in_channels=3,
            out_channels=32,
            kernel_size=3,
            padding=1
        ),
        nn.ReLU(),

        nn.MaxPool2d(kernel_size=2),

        # 32 x 32 -> 16 x 16

        nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            padding=1
        ),
        nn.ReLU(),

        nn.MaxPool2d(kernel_size=2),

        # 16 x 16 -> 8 x 8

        nn.Flatten(),

        nn.Linear(64 * 8 * 8, 128),
        nn.ReLU(),

        nn.Linear(128, 10)
    )



