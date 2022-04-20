import torch
import numpy as np
import random


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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


def sample_evaluate_noise(args, cat_idx=0, con_idx=0):
    # sample latent code
    fixed_z = torch.randn(args.latent_dim)
    list_z = []
    for i in range(args.category_number):
        latent_cat = torch.zeros((args.discrete_var, args.category_number))
        latent_cat[cat_idx, i] = 1
        latent_cont = torch.zeros((args.continuous_var,))
        c_range = torch.linspace(start=-2, end=2, steps=10)
        for k in range(10):
            latent_cont[con_idx] = c_range[k]
            list_z.append(torch.cat([fixed_z, latent_cat.view(-1, 1).squeeze(), latent_cont]).unsqueeze(0))
    return torch.cat(list_z, dim=0)