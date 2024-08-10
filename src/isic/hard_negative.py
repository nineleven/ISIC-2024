import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from PIL import Image
import cv2

import h5py
import io

import torch
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.models import squeezenet1_1

from sklearn.metrics import roc_curve, auc

import pytorch_lightning as pl


class CollateFn:

    def __init__(self, augment=True):
        self.augment = augment

    def __call__(self, batch):
        xs = []
        ys = []

        for x, y in batch:
            x = cv2.resize(x, (150, 150), interpolation=cv2.INTER_CUBIC)

            if np.random.random() < 0.5:
                x = np.flip(x, axis=0)

            if np.random.random() < 0.5:
                x = np.flip(x, axis=1)

            xs.append(x)
            ys.append(y)

        xs = (np.asarray(xs).astype("float32") / 255).astype("float16")
        ys = np.eye(2)[ys].astype("float16")

        return xs, ys


class ISIC2024Dataset(Dataset):

    def __init__(self, path: str, target: pd.Series, ids: list[str] = None):
        super().__init__()
        if ids is None:
            with h5py.File(path, 'r') as f:
                ids = list(f.keys())
        self.path = path
        self.target = target
        self.ids = ids
        self.file = h5py.File(self.path, 'r')

    def __getitem__(self, idx):
        isic_id = self.ids[idx]

        img_raw_data = self.file[isic_id][()]

        img = np.array(Image.open(io.BytesIO(img_raw_data)))
        label = self.target.loc[isic_id]

        assert img.shape[0] == img.shape[1]

        return img, label

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


class ISIC2024Model(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = squeezenet1_1(num_classes=2)

    def _preprocess_batch(self, batch):
        x, y = batch
        x = torch.as_tensor(np.moveaxis(x, -1, 1), device=self.device)
        y = torch.as_tensor(y, device=self.device)
        return x, y

    def training_step(self, batch, batch_idx):
        x, y = self._preprocess_batch(batch)
        y_pred_logits = self.model(x)

        y_mean = y[:, 1].mean()
        weights = torch.as_tensor([1, y_mean / (1 - y_mean)], dtype=torch.float16, device=self.device)
        loss = F.cross_entropy(y_pred_logits, y, weights)

        self.log("train_loss", loss.item(), prog_bar=True, batch_size=x.shape[0])
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = self._preprocess_batch(batch)
        y_pred_logits = self.model(x)

        y_mean = y[:, 1].mean()
        weights = torch.as_tensor([1, y_mean / (1 - y_mean)], dtype=torch.float16, device=self.device)
        loss = F.cross_entropy(y_pred_logits, y, weights)
        acc = self.compute_balanced_accuracy(y_pred_logits, y)
        pauc = self.compute_pauc_score(y_pred_logits, y)

        self.log("val_loss", loss.item(), prog_bar=True, batch_size=x.shape[0])
        self.log("val_acc", acc.item(), prog_bar=True, batch_size=x.shape[0])
        self.log("val_pauc", pauc.item(), prog_bar=True, batch_size=x.shape[0])
        return loss

    def compute_balanced_accuracy(self, y_pred_logits, y):
        y_pred_proba = F.softmax(y_pred_logits, dim=1)
        y_pred = torch.argmax(y_pred_proba, dim=1)

        P = y[:, 1].sum()
        N = (1 - y[:, 1]).sum()

        TP = y_pred[y[:, 1] == 1].sum()
        TN = (1 - y_pred)[y[:, 1] == 0].sum()

        if P == 0:
            return TN / N

        if N == 0:
            return TP / P

        return (TN / N + TP / P) / 2

    def compute_pauc_score(self, y_pred_logits, y):
        y_pred_proba = F.softmax(y_pred_logits, dim=1)
        fpr, tpr, _ = roc_curve(y[:, 1].cpu().numpy(), y_pred_proba[:, 1].cpu().numpy())
        am = np.argmax(tpr >= 0.8)
        pauc = auc(fpr[am:], tpr[am:] - tpr[am])
        return torch.as_tensor(pauc, dtype=torch.float16, device=y.device)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), 1e-4)
