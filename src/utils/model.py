import torch
import numpy as np

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


def sample_latent(args):
    # Sample Z from N(0,1)
    z = torch.randn((args.batch_size, args.latent_dim), device=args.device)

    # Sample categorical variable
    idx = np.zeros((args.discrete_var, args.batch_size))
    latent_cat = torch.zeros((args.batch_size, args.discrete_var, args.category_number), device=args.device)

    for i in range(args.discrete_var):
        idx[i] = np.random.randint(args.category_number, size=args.batch_size)
        latent_cat[torch.arange(0, args.batch_size), i, idx[i]] = 1.0

    # Sample continuous variable
    latent_cont = torch.rand((args.batch_size, args.continuous_var), device=args.device) * 2 - 1

    return torch.cat([z, latent_cat.view(args.batch_size, -1), latent_cont], dim=1), idx
