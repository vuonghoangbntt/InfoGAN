import os

import torch
from ..model import Generator, Discriminator
from ..utils.model import sample_evaluate_noise
from ..utils.visualize import plot_image_grid


class InfoGANInfer:
    def __init__(self, args):
        saved_model = self.load_model()
        self.args = args
        self.model_args = saved_model['args']
        self.base_name = f'InfoGAN_gen-lr={self.model_args.generator_learning_rate}_dis-lr={self.model_args.discriminator_learning_rate}' \ 
                         f'_epoch={self.model_args.num_epochs}_dist-weight={self.model_args.dist_weight}_cont-weight={self.model_args.cont_weight}'
        self.save_path = os.path.join(args.output_dir, self.base_name)
        self.generator = Generator(self.args).to(self.args.device)
        self.generator.load_state_dict(saved_model['generator'])
        self.discriminator = Discriminator(self.args).to(self.args.device)
        self.discriminator.load_state_dict(saved_model['discriminator'])
        self.loss_dict = saved_model['loss']

    def load_model(self):
        saved_model = torch.load(self.args.model_path)
        return saved_model

    def infer(self):
        for cat_idx in range(self.model_args.discrete_var):
            for con_idx in range(self.args.continuous_var):
                z = sample_evaluate_noise(cat_idx, con_idx).to(self.args.device)
                output_file = os.path.join(self.save_path, f'Discrete-c{cat_idx}_Continuous-c{con_idx}.png')
                fake_img = self.generator(z)
                plot_image_grid(fake_img, title=f'Varying discrete var c{cat_idx} and continuous var c{con_idx}',
                                y_axis=f'Discrete Var c{cat_idx}', x_axis=f'Continuous Var c{con_idx}',
                                save_figure=self.args.save_figure, figure_dir=output_file)
