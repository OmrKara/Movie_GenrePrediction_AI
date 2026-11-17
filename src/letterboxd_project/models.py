from torch import nn


class ProjectMLP(nn.Module):
    """Two-hidden-layer multilabel MLP used for the U7/U8 project stage."""

    def __init__(self, input_dim=1000, hidden=(128, 64), dropout=0.2, output_dim=5):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden[0]), nn.ReLU()]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.extend([nn.Linear(hidden[0], hidden[1]), nn.ReLU()])
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden[1], output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
