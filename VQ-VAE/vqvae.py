import os
import torch
import torch.nn as nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from torch.optim import Adam
import torch.nn.functional as F
from functions import vq, vq_st
from utils import *
import logging
logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")


def init_weights(module: nn.Module):
    """
    Initializes the weights of any convolutional layer with the xavier uniform init.
    """
    if "Conv" in module.__class__.__name__:
        nn.init.xavier_uniform_(module.weight.data)
        module.bias.data.fill_(0)


class Codebook(nn.Module):
    """
    Codebook implementation for the mapping from latent representation of encoder to closest codebook vectors.
    """
    def __init__(self, num_vectors, dim_vectors):
        super(Codebook, self).__init__()
        self.codebook = nn.Embedding(num_vectors, dim_vectors)

    def forward(self, z):
        """
        Find the closest matching codebook vector for every vector in z using euclidean distance.
        :param z: latent representation of input
        :return: mapped latent representation to codebook vectors
        """
        z_e = z.permute(0, 2, 3, 1)
        codes, indices = vq_st(z_e, self.codebook.weight.detach())
        codes = codes.permute(0, 3, 1, 2)

        codes_bar = torch.index_select(self.codebook.weight, dim=0, index=indices)\
            .view_as(z_e)\
            .permute(0, 3, 1, 2)
        return codes, codes_bar


class ResnetBlock(nn.Module):
    """
    ResNet-Block from the original ResNet paper.
    """
    def __init__(self, dim):
        super(ResnetBlock, self).__init__()
        self.block = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(dim, dim, 3, 1, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.Conv2d(dim, dim, 3, 1, 1),
            nn.BatchNorm2d(dim)
        )

    def forward(self, x):
        return x + self.block(x)


class Encoder(nn.Module):
    """
    Encode image into latent space --> encoding dim = codebook_dim x codebook_dim. 32x32
    Input image: 3x256x256 (256 -> 128 -> 64 -> 32)
    """
    def __init__(self, dim=64):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, dim, 4, 2, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.Conv2d(dim, dim, 4, 2, 1),
            ResnetBlock(dim),
            ResnetBlock(dim)
        )
        self.apply(init_weights)

    def forward(self, x):
        return self.encoder(x)


class Decoder(nn.Module):
    """
    Decode image from latent space to original resolution.
    """
    def __init__(self, dim):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            ResnetBlock(dim),
            ResnetBlock(dim),
            nn.ReLU(True),
            nn.ConvTranspose2d(dim, dim, 4, 2, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.ConvTranspose2d(dim, 1, 4, 2, 1),
            nn.Tanh()
        )
        self.apply(init_weights)

    def forward(self, x):
        return self.decoder(x)


class VQVAE(nn.Module):
    """
    Main class for the Vector Quantization VAE.
    """
    def __init__(self, encoder, decoder, codebook_length=512, codebook_dim=256):
        super(VQVAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.codebook = Codebook(codebook_length, codebook_dim)
        self.training = True
        self.beta = 0.25

    def forward(self, x):
        z_e = self.encoder(x)
        z_q, z_q_bar = self.codebook(z_e)
        x_rec = self.decoder(z_q)
        commit_loss = 0.
        embedding_loss = 0.
        if self.training:
            commit_loss = self.beta * F.mse_loss(z_q_bar.detach(), z_e)
            embedding_loss = F.mse_loss(z_q_bar, z_e.detach())
        return x_rec, embedding_loss, commit_loss


def train(config):
    if not os.path.exists("results"):
        os.mkdir("results")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    enc = Encoder(config["dim"])
    dec = Decoder(config["dim"])
    model = VQVAE(enc, dec).to(device)
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    mnist = datasets.MNIST(config["data_path"], train=True, download=True, transform=transform)
    train_data = DataLoader(mnist, 1, shuffle=False, pin_memory=True)
    optimizer = Adam(model.parameters(), 2e-4)
    best_loss = 1e100
    for e in range(config["epochs"]):
        logging.info(f"Starting epoch {e}!")
        for i, (X, _) in enumerate(train_data, 1):
            optimizer.zero_grad()
            X = X.to(device)
            logging.debug(f"Starting with {i}. batch.")
            X_rec, embedding_loss, commit_loss = model(X)
            rec_loss = F.mse_loss(X, X_rec)
            loss = rec_loss + embedding_loss + commit_loss
            loss.backward()
            optimizer.step()
            if i % 1000 == 0:
                logging.info(f"Loss: {loss}. ({rec_loss}, {embedding_loss}, {commit_loss})")
                save_fig(X, X_rec, epoch=e, step=i)

            if loss < best_loss:
                torch.save(model.state_dict(), os.path.join("results", "_best_model.pt"))


if __name__ == '__main__':
    config = {"epochs": 10,
              "dim": 256,
              "data_path": os.path.join("pytorch-vqvae", "data")
              }
    train(config)
