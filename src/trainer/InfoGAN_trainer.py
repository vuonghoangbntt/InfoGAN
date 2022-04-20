from .trainer import Trainer
import numpy as np
import logging
import itertools
from tqdm import tqdm
import torch
from torch.optim import Adam
import torch.nn as nn
import os

from ..model.module import NLL_gaussian
from ..utils.model import sample_latent


class InfoGANTrainer(Trainer):
    def __init__(self, args):
        super(InfoGANTrainer, self).__init__()
        self.args = args
        self.base_name = f'InfoGAN_gen-lr={args.generator_learning_rate}_dis-lr={args.discriminator_learning_rate}' \
                         f'_epoch={args.num_epochs}_dist-weight={args.dist_weight}_cont-weight={args.cont_weight}'
        self.save_path = os.path.join(args.output_dir, self.base_name)

        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        # Init Logger
        self.logger = logging.getLogger()
        file_handler = logging.FileHandler("{0}/{1}".format(self.save_path, "log"),
                                           mode='w')
        self.logger.addHandler(file_handler)

    def train(self, generator, discriminator, train_loader, test_loader):

        # Optimizer prepare
        generator_optimizer = Adam(itertools.chain(*[generator.parameters(), discriminator.Q_head.parameters(),
                                                     discriminator.latent_cont_mu.parameters(),
                                                     discriminator.latent_disc.parameters()]),
                                   lr=self.args.generator_learning_rate, betas=(0.5, 0.999))
        discriminator_optimizer = Adam(
            itertools.chain(*[discriminator.module_shared.parameters(), discriminator.discriminator_head.parameters()]),
            lr=self.args.discriminator_learning_rate, betas=(0.5, 0.999))

        # Loss function
        adversarial_loss = nn.BCELoss()
        categorical_loss = nn.CrossEntropyLoss()
        continuous_loss = NLL_gaussian()

        # Training
        loss_dict = {
            'dis_loss': [],
            'info_loss': [],
            'gen_loss': [],
            'discrete_loss': [],
            'continuous_loss': []
        }
        for epoch in range(self.args.num_epochs):
            self.logger.info("-------------------------------------------")
            self.logger.info(f"|             Epoch {epoch}              |")
            self.logger.info("-------------------------------------------")
            loss_epoch = self.train_epoch(generator, discriminator, generator_optimizer, discriminator_optimizer,
                                          train_loader, adversarial_loss, categorical_loss, continuous_loss)
            for key in loss_dict.keys():
                loss_dict[key].append(loss_epoch[key])
            self.save_model(generator, discriminator, loss_dict)

    def train_epoch(self, generator, discriminator, generator_optimizer, discriminator_optimizer,
                    train_loader, adversarial_loss, categorical_loss, continuous_loss):
        loss_dict = {
            'dis_loss': [],
            'info_loss': [],
            'gen_loss': [],
            'discrete_loss': [],
            'continuous_loss': []
        }
        step = 0
        for i, (batch, _) in tqdm(enumerate(train_loader)):
            if batch.size()[0] != self.args.batch_size:
                self.args.batch_size = batch.size()[0]

            data = batch.to(self.args.device)
            real_label = torch.ones((batch.size()[0],), device=self.args.device)
            fake_label = torch.zeros((batch.size()[0],), device=self.args.device)

            # Discriminator training
            discriminator_optimizer.zero_grad()

            prob_real, _, _, _ = discriminator(data)
            real_loss = adversarial_loss(prob_real, real_label)
            real_loss.backward()

            z, idx = sample_latent(self.args)
            fake_data = generator(z)
            prob_fake_D, _, _, _ = discriminator(fake_data.detach())
            fake_loss = adversarial_loss(prob_fake_D, fake_label)
            fake_loss.backward()

            loss_D = real_loss + fake_loss
            loss_dict['dis_loss'].append(loss_D.item())
            discriminator_optimizer.step()

            # Generator training
            generator_optimizer.zero_grad()
            prob_fake, cat_prob, mu, var = discriminator(fake_data)
            loss_G = adversarial_loss(prob_fake, real_label)
            loss_dict['gen_loss'].append(loss_G.item())

            # Discrete variable loss
            target = torch.LongTensor(idx).to(self.args.device)
            loss_c_dist = 0
            for j in range(self.args.discrete_var):
                loss_c_dist += categorical_loss(cat_prob[:, j, :], target[j, :])
            loss_c_dist = loss_c_dist * self.args.dist_weight
            loss_dict['discrete_loss'].append(loss_c_dist.item())

            # Continuous variable loss
            loss_c_cont = continuous_loss(
                z[:, self.args.latent_dim + self.args.category_number * self.args.discrete_var:], mu,
                var).mean(0)
            loss_c_cont = loss_c_cont * self.args.cont_weight
            loss_dict['continuous_loss'].append(loss_c_cont.detach().cpu())

            loss_info = loss_G + loss_c_dist + loss_c_cont.sum()
            loss_dict['info_loss'].append(loss_info.item())
            loss_info.backward()
            generator_optimizer.step()

            if step % 100 == 0:
                self.logger.info(
                    f'Step {step}: Discriminator loss: {loss_D.item():.3f}\tInfo loss: {loss_info.item():.3f}')
                self.logger.info(
                    f'Generator loss: {loss_G.item():.3f}\tDis loss: {loss_c_dist.item():.3f}\tCont loss: {loss_c_cont.sum().item():.3f}')
            step += 1
        return loss_dict

    def eval(self):
        self.logger.info("EVALUATION NOT SUPPORTED YET")

    def save_model(self, generator, discriminator, loss_dict):
        file_path = os.path.join(self.save_path, 'model.pt')
        torch.save({
            'generator': generator.state_dict(),
            'discriminator': discriminator.state_dict(),
            'loss': loss_dict,
            'args': self.args
        }, file_path)
