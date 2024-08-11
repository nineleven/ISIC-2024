import io

import albumentations as A
import cv2
import h5py
import numpy as np
import pytorch_lightning as pl
import torch

from PIL import Image
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader

from .model import ResNetAutoEncoder, get_configs

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


class Autoencoder(pl.LightningModule):

    def __init__(self):
        super().__init__()
        configs, bottleneck = get_configs("resnet18")
        self.ae = ResNetAutoEncoder(configs, bottleneck)

    def training_step(self, batch, batch_idx):
        x = torch.as_tensor(np.moveaxis(batch, -1, 1), device=self.device)
        x_hat = self.ae(x)

        l2_loss = F.mse_loss(x_hat, x)
        l1_loss = F.l1_loss(x_hat, x)

        self.log("train_l1_loss", l1_loss.item(), batch_size=batch.shape[0])
        self.log("train_l2_loss", l2_loss.item(), prog_bar=True, batch_size=batch.shape[0])

        return l1_loss

    def validation_step(self, batch, batch_idx):
        x = torch.as_tensor(np.moveaxis(batch, -1, 1), device=self.device)
        x_hat = self.ae(x)

        if batch_idx % 100 == 99:
            self.log_images(x, x_hat)

        l2_loss = F.mse_loss(x_hat, x)
        l1_loss = F.l1_loss(x_hat, x)

        self.log("val_l1_loss", l1_loss.item(), batch_size=batch.shape[0])
        self.log("val_l2_loss", l2_loss.item(), prog_bar=True, batch_size=batch.shape[0])

        return l1_loss

    def log_images(self, x, x_hat_sigmoid):
        idx = np.random.randint(0, x.shape[0])

        img1 = np.moveaxis(x[idx].cpu().detach().numpy(), 0, -1)
        img2 = np.moveaxis(x_hat_sigmoid[idx].cpu().detach().numpy(), 0, -1)

        self.logger.log_image("samples", [img1, img2])

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), 1e-3)
        # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, eta_min=5e-5)
        # return [optimizer], [lr_scheduler]
        return optimizer
