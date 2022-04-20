import torch
import torch.nn as nn
from .module import Reshape


# Discriminator
class Discriminator(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.module_shared = nn.Sequential(
            nn.Conv2d(in_channels=1,
                      out_channels=64,
                      kernel_size=4,
                      stride=2,
                      padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(in_channels=64,
                      out_channels=128,
                      kernel_size=4,
                      stride=2,
                      padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            Reshape(-1, 128 * 7 * 7),
            nn.Linear(in_features=128 * 7 * 7, out_features=1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
        )
        # Discriminator head
        self.discriminator_head = nn.Sequential(
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )

        # Q head
        self.Q_head = nn.Sequential(
            nn.Linear(1024, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1, inplace=True)
        )

        # discrete latent
        self.latent_disc = nn.Sequential(
            nn.Linear(128, args.discrete_var * args.category_number),
            Reshape(-1, args.discrete_var, args.category_number),
            nn.Softmax(dim=2)
        )

        # Continuous mean and variance
        self.latent_cont_mu = nn.Linear(128, args.continuous_var)
        self.latent_cont_var = nn.Linear(128, args.continuous_var)

    def forward(self, x):
        out = self.module_shared(x)
        dis_prob = self.discriminator_head(out).squeeze()
        internal_Q = self.Q_head(out)
        c_disc_logits = self.latent_disc(internal_Q)
        c_cont_mu = self.latent_cont_mu(internal_Q)
        c_cont_var = torch.exp(self.latent_cont_var(internal_Q))
        return dis_prob, c_disc_logits, c_cont_mu, c_cont_var

