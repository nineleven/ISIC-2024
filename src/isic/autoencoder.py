import io

import albumentations as A
import cv2
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import wandb

from PIL import Image
from kaggle_secrets import UserSecretsClient
from pytorch_lightning.loggers.wandb import WandbLogger
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader


class ISIC2024Dataset(Dataset):

    def __init__(self, path: str, ids: list[str] = None):
        super().__init__()
        if ids is None:
            with h5py.File(path, 'r') as f:
                ids = list(f.keys())
        self.path = path
        self.ids = ids
        self.file = h5py.File(self.path, 'r')

        self.transform = A.Compose([A.RandomCrop(128, 128)])

    def __getitem__(self, idx):
        isic_id = self.ids[idx]

        img_raw_data = self.file[isic_id][()]
        img = np.array(Image.open(io.BytesIO(img_raw_data)))

        assert img.shape[0] == img.shape[1]

        if img.shape[0] < 128:
            img = cv2.resize(img, (128, 128), interpolation=cv2.INTER_CUBIC)
        img = self.transform(image=img)["image"]
        img = (img.astype("float32") / 255).astype("float16")

        return img

    def __len__(self):
        return len(self.ids)

    def __del__(self):
        if hasattr(self, "file"):
            self.file.close()

    def __getstate__(self):
        return {k: v for k, v in self.__dict__.items() if k != "file"}

    def __setstate__(self, state):
        import h5py
        self.__dict__.update(state)
        self.file = h5py.File(self.path, 'r')


class Encoder(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=7, padding=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=5, padding=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=2),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=2),
            nn.ReLU(),
        )
        self.fc = nn.Linear(4 * 4 * 256, 512)

    def forward(self, x):
        z1 = self.conv(x).reshape(-1, 256 * 4 * 4)
        z2 = self.fc(z1)
        return z2


class Decoder(nn.Module):

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(512, 4 * 4 * 256)
        self.conv = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ReLU(),
            nn.Conv2d(256, 128, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.ReLU(),
            nn.Conv2d(32, 16, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.ReLU(),
            nn.Conv2d(16, 3, kernel_size=3, padding=1, stride=1),
        )

    def forward(self, z):
        x1 = self.fc(z).reshape(-1, 256, 4, 4)
        x2 = self.conv(x1)
        return x2


class Autoencoder(pl.LightningModule):

    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat

    def training_step(self, batch, batch_idx):
        x = torch.as_tensor(np.moveaxis(batch, -1, 1), device=self.device)
        x_hat = self(x)
        x_hat_sigmoid = F.sigmoid(x_hat)

        l2_loss = F.mse_loss(x_hat_sigmoid, x)
        l1_loss = F.l1_loss(x_hat_sigmoid, x)
        log_loss = F.binary_cross_entropy_with_logits(x_hat, x)

        self.log("train_l1_loss", l1_loss.item(), batch_size=batch.shape[0])
        self.log("train_l2_loss", l2_loss.item(), prog_bar=True, batch_size=batch.shape[0])
        self.log("train_log_loss", log_loss.item(), batch_size=batch.shape[0])
        return l2_loss

    def validation_step(self, batch, batch_idx):
        x = torch.as_tensor(np.moveaxis(batch, -1, 1), device=self.device)
        x_hat = self(x)
        x_hat_sigmoid = F.sigmoid(x_hat)

        if batch_idx % 50 == 49:
            self.log_images(x, x_hat_sigmoid)

        l2_loss = F.mse_loss(x_hat_sigmoid, x)
        l1_loss = F.l1_loss(x_hat_sigmoid, x)
        log_loss = F.binary_cross_entropy_with_logits(x_hat, x)

        self.log("val_l1_loss", l1_loss.item(), batch_size=batch.shape[0])
        self.log("val_l2_loss", l2_loss.item(), prog_bar=True, batch_size=batch.shape[0])
        self.log("val_log_loss", log_loss.item(), batch_size=batch.shape[0])

        return l2_loss

    def log_images(self, x, x_hat_sigmoid):
        idx = np.random.randint(0, x.shape[0])

        img1 = np.moveaxis(x[idx].cpu().detach().numpy(), 0, -1)
        img2 = np.moveaxis(x_hat_sigmoid[idx].cpu().detach().numpy(), 0, -1)

        self.logger.log_image("samples", [img1, img2])

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), 1e-4)