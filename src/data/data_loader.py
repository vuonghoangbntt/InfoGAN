import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms


def load_MNIST_dataset(args):
    mnist_transform = transforms.Compose([
        transforms.Resize(args.image_size),
        transforms.CenterCrop(args.image_size),
        transforms.ToTensor()])
    trainset = torchvision.datasets.MNIST('MNIST/processed/training.pt', download=True, transform=mnist_transform,
                                          train=True)
    testset = torchvision.datasets.MNIST('MNIST/processed/testing.pt', download=True, transform=mnist_transform,
                                         train=False)
    train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(testset, batch_size=args.batch_size, shuffle=False)
    return train_loader, test_loader
