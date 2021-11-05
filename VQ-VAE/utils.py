import os
import matplotlib.pyplot as plt


def save_fig(X_true, X_rec, epoch, step):
    """
    Method for saving actual images and generated images for comparison.
    :param X_true: Ground truth images.
    :param X_rec: Reconstructed images from model.
    :param epoch: Current epoch.
    :param step: Current step of epoch.
    """
    X_true = X_true.cpu().detach()
    X_rec = X_rec.cpu().detach()
    num_images = X_true.size(0)
    for i in range(num_images):
        plt.subplot(num_images, 2, i+1)
        plt.axis("off")
        plt.imshow(X_true[i, 0, :, :], cmap="gray")
        plt.subplot(num_images, 2, i+2)
        plt.axis("off")
        plt.imshow(X_rec[i, 0, :, :], cmap="gray")
    plt.savefig(os.path.join("results", f"{epoch}_{step}_images"))
    plt.close()
