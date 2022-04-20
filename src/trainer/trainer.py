from abc import abstractmethod


class Trainer:
    def __init__(self):
        pass

    @abstractmethod
    def train(self):
        NotImplemented

    @abstractmethod
    def eval(self):
        NotImplemented

    @abstractmethod
    def train_epoch(self):
        NotImplemented

    @abstractmethod
    def save_model(self):
        NotImplemented