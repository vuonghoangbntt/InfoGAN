import argparse
from src.model import Generator, Discriminator
from src.utils import init_logger, reset_logger, weights_init_normal, sample_latent
from src.trainer.InfoGAN_trainer import InfoGANTrainer
from src.data import load_MNIST_dataset


def main(args):
    init_logger()

    generator = Generator(args)
    discriminator = Discriminator(args)
    train_loader, test_loader = load_MNIST_dataset(args)

    if args.do_train:
        trainer = InfoGANTrainer(args)
        trainer.train(generator, discriminator, train_loader, test_loader)
    if args.do_infer:
        pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", default='mnist', type=str, help="Dataset use to train model")
    parser.add_argument("--output_dir", default='./experiment/', type=str, help="Path to save model")
    parser.add_argument("--image_size", default=28, type=int, help="Input image size")
    parser.add_argument("--batch_size", default=128, type=int, help="Batch size when training")
    parser.add_argument("--num_epochs", default=20, type=int, help="Num training epochs")
    parser.add_argument("--generator_learning_rate", default=0.001, type=int, help="Generator learning rate")
    parser.add_argument("--discriminator_learning_rate", default=0.0002, type=int, help="Discriminator learning rate")
    parser.add_argument("--continuous_var", default=2, type=int, help="Number of continuous variable")
    parser.add_argument("--discrete_var", default=1, type=int, help="Number of discrete variable")
    parser.add_argument("--latent_dim", default=62, type=int, help="Latent variable dimension")
    parser.add_argument("--category_number", default=10, type=int, help="Number of discrete category")
    parser.add_argument("--dist_weight", default=1.0, type=float, help="Discrete loss weight")
    parser.add_argument("--cont_weight", default=0.1, type=float, help="Continuous continuous weight")
    parser.add_argument("--device", default="cpu", type=str, help="Training device (cuda or cpu)")
    parser.add_argument("--do_train", action="store_true", help="Whether or not do training")
    parser.add_argument("--do_infer", action="store_true", help="Whether or not do inference")

    args = parser.parse_args()
    main(args)