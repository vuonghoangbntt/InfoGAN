import matplotlib.pyplot as plt
import numpy as np
import torchvision


def plot_lines(x, lines, labels, title='', x_axis='', y_axis='', fig_size=(10, 5)):
    plt.figure(figsize=fig_size)

    for line, label in zip(lines, labels):
        plt.plot(x, line, label=label)

    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    plt.title(title)
    plt.legend()
    plt.show()


def plot_image_grid(img, title='', x_axis='', y_axis='', nrow=10, padding=2, fig_size=(8, 8), save_figure=False, figure_dir=None):
    plt.figure(figsize=fig_size)

    plt.imshow(np.transpose(torchvision.utils.make_grid(
        img.detach().cpu(), nrow=nrow, padding=padding, normalize=True), (1, 2, 0)))

    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    plt.title(title)
    if save_figure and figure_dir is not None:
        plt.savefig(figure_dir)
    plt.show()
