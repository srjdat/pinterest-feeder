import torch.nn as nn

class PinterestFeeder(nn.Module):
    def __init__(self, input_size: int) -> None:
        super().__init__()
        self.layer1 = nn.Linear(input_size, 256)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(256, 1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        return self.layer2(x)
