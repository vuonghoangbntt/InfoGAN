import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.args = args

        input_dim = args.latent_dim + args.continuous_var + args.discrete_var*args.category_number

        self.linear1 = nn.Linear(input_dim, 1024, bias=False)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(1024)

        self.linear2 = nn.Linear(1024, 7 * 7 * 128, bias=False)
        self.bn2 = nn.BatchNorm1d(7 * 7 * 128)

        self.conv1 = nn.ConvTranspose2d(128, 64, 4, 2, 1)
        self.bn3 = nn.BatchNorm2d(64)

        self.conv2 = nn.ConvTranspose2d(64, 1, 4, 2, 1)

    def forward(self, x):
        x = self.relu(self.bn1(self.linear1(x)))
        x = self.relu(self.bn2(self.linear2(x)))
        x = x.view(-1, 128, 7, 7)
        x = self.relu(self.bn3(self.conv1(x)))
        x = torch.tanh(self.conv2(x))
        return x
