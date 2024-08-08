import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from PIL import Image

import h5py
import io

import torch
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.models import squeezenet1_1

from sklearn.metrics import roc_curve, auc

import pytorch_lightning as pl

import multiprocess as multiprocessing

from .augmentations import ImageAugmenter


class CollateFn:

    def __init__(self, augment=True):
        self.augment = augment

    def __call__(self, batch):
        max_size = max(x.shape[0] for x, _ in batch)

        xs = []
        ys = []

        for x, y in batch:
            size = x.shape[0]

            if size != max_size:
                p1 = (max_size - size) // 2
                p2 = (max_size - size) - p1
                x = np.pad(x, [(p1, p2), (p1, p2), (0, 0)])

                if self.augment and np.random.random() < 0.5:
                    aug = ImageAugmenter()
                    if np.random.random() < 0.5:
                        x = aug.random_rotation_flip(x)
                    if np.random.random() < 0.5:
                        x = aug.random_scale(x)
                    if np.random.random() < 0.5:
                        x = aug.random_shear(x)

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


class ISIC2024BalancedDataset(Dataset):

    def __init__(self, ds_pos, ds_neg, pos_freq):
        self.ds_pos = ds_pos
        self.ds_neg = ds_neg

        self.pos_freq = pos_freq

    def __getitem__(self, idx):
        if idx % self.pos_freq == self.pos_freq - 1:
            return self.ds_pos[(idx // self.pos_freq) % len(self.ds_pos)]
        return self.ds_neg[idx - idx // self.pos_freq]

    def __len__(self):
        return len(self.ds_neg) + len(self.ds_neg) // self.pos_freq


class ISIC2024DataLoader:

    def __init__(self, dataset: ISIC2024Dataset, batch_size: int, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.k = 5

    def _make_batches(self, items):
        items = sorted(items, key=lambda x: x[0].shape[0])

        for i in range(0, len(items), self.batch_size):
            j = min(len(items), i + self.batch_size)
            max_size = items[j - 1][0].shape[0]

            xs = []
            ys = []

            for x, y in items[i: j]:
                size = x.shape[0]

                if size != max_size:
                    p1 = (max_size - size) // 2
                    p2 = (max_size - size) - p1
                    x = np.pad(x, [(p1, p2), (p1, p2), (0, 0)])

                xs.append(x)
                ys.append(y)

            xs = (np.asarray(xs).astype("float32") / 255).astype("float16")
            ys = np.eye(2)[ys].astype("float16")

            yield xs, ys

    def load(self, queue, indices):
        items = []

        for idx in indices:
            x, y = self.dataset[idx]
            items.append((x, y))

            if len(items) == self.k * self.batch_size:
                for batch in self._make_batches(items):
                    queue.put(batch)
                items = []

            while queue.qsize() > 10:
                continue

        if len(items) > 0:
            for batch in self._make_batches(items):
                queue.put(batch)

    def __iter__(self):
        indices = np.arange(len(self.dataset))

        if self.shuffle:
            np.random.shuffle(indices)

        with multiprocessing.Manager() as manager:
            queue = manager.Queue()
            with multiprocessing.Pool(2) as pool:
                pool.map_async(self.load, [(self, queue, indices[::2]), (self, queue, indices[1::2])])

                try:
                    while True:
                        yield queue.get(timeout=10)
                except Exception as ex:
                    print(ex)

    def __len__(self):
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size


class ISIC2024Model(pl.LightningModule):

    def __init__(self, pos_freq):
        super().__init__()
        self.model = squeezenet1_1(num_classes=2, dropout=0.75)
        self.pos_freq = pos_freq

    def _preprocess_batch(self, batch):
        x, y = batch
        x = torch.as_tensor(np.moveaxis(x, -1, 1), device=self.device)
        y = torch.as_tensor(y, device=self.device)
        return x, y

    def training_step(self, batch, batch_idx):
        x, y = self._preprocess_batch(batch)
        y_pred_logits = self.model(x)

        weights = torch.as_tensor([1, self.pos_freq - 1], dtype=torch.float16, device=self.device)
        loss = F.cross_entropy(y_pred_logits, y, weights)

        self.log("train_loss", loss.item(), prog_bar=True, batch_size=x.shape[0])
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = self._preprocess_batch(batch)
        y_pred_logits = self.model(x)

        weights = torch.as_tensor([1, self.pos_freq - 1], dtype=torch.float16, device=self.device)
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



if __name__ == "__main__":
    train_metadata = pd.read_csv("/kaggle/input/isic-2024-challenge/train-metadata.csv")
    train_metadata.head()

    val_ratio = 0.3
    batch_size = 128
    pos_freq = 5

    all_patient_ids = train_metadata["patient_id"].unique()
    num_patients = len(all_patient_ids)

    num_train_patients = int(num_patients * (1 - val_ratio))
    num_val_patients = num_patients - num_train_patients

    perm = np.random.permutation(all_patient_ids)
    train_patients = perm[:num_train_patients]
    val_patients = perm[num_train_patients:]

    train_pos_indices = train_metadata.loc[
        train_metadata["patient_id"].isin(train_patients) & (train_metadata["target"] == 1), "isic_id"].tolist()
    train_neg_indices = train_metadata.loc[
        train_metadata["patient_id"].isin(train_patients) & (train_metadata["target"] == 0), "isic_id"].tolist()

    val_pos_indices = train_metadata.loc[
        train_metadata["patient_id"].isin(val_patients) & (train_metadata["target"] == 1), "isic_id"].tolist()
    val_neg_indices = train_metadata.loc[
        train_metadata["patient_id"].isin(val_patients) & (train_metadata["target"] == 0), "isic_id"].tolist()

    train_pos_ds = ISIC2024Dataset("/kaggle/input/isic-2024-challenge/train-image.hdf5",
                                   train_metadata.set_index("isic_id")["target"], train_pos_indices)
    train_neg_ds = ISIC2024Dataset("/kaggle/input/isic-2024-challenge/train-image.hdf5",
                                   train_metadata.set_index("isic_id")["target"], train_neg_indices)

    val_pos_ds = ISIC2024Dataset("/kaggle/input/isic-2024-challenge/train-image.hdf5",
                                 train_metadata.set_index("isic_id")["target"], val_pos_indices)
    val_neg_ds = ISIC2024Dataset("/kaggle/input/isic-2024-challenge/train-image.hdf5",
                                 train_metadata.set_index("isic_id")["target"], val_neg_indices)

    train_ds = ISIC2024BalancedDataset(train_pos_ds, train_neg_ds, pos_freq)
    val_ds = ISIC2024BalancedDataset(val_pos_ds, val_neg_ds, pos_freq)

    print("train ds size:", len(train_ds))
    print("val ds size:", len(val_ds))

    multiprocessing.set_start_method("spawn")

    train_loader = DataLoader(train_ds, batch_size=batch_size, num_workers=2, collate_fn=CollateFn())  # ISIC2024DataLoader(train_ds, batch_size)
    print("train loader:", len(train_loader), "batches")
    val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=2, collate_fn=CollateFn(augment=False))  # ISIC2024DataLoader(val_ds, batch_size)
    print("train loader:", len(val_loader), "batches")

    model = ISIC2024Model(pos_freq)

    trainer = pl.Trainer(precision="16-mixed")
    trainer.fit(model, train_loader, val_loader)