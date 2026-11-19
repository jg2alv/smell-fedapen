"""Client-local Fashion-MNIST partitions and supervised pair sampling."""

import math
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose, Normalize, ToTensor


TRANSFORMS = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])


def transform_images(batch):
    batch["image"] = [TRANSFORMS(image) for image in batch["image"]]
    return batch


def seed_worker(worker_id):
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)


class PairDataset(Dataset):
    def __init__(self, partition, num_pairs, seed):
        self.partition = partition.with_transform(transform_images)
        self.num_pairs = num_pairs
        self.seed = seed
        self.epoch = 0
        labels = np.asarray(partition["label"])
        self.indices = {
            int(label): np.flatnonzero(labels == label).tolist()
            for label in np.unique(labels)
        }
        self.labels = list(self.indices)
        self.positive_labels = [label for label in self.labels if len(self.indices[label]) > 1]

    def __len__(self):
        return self.num_pairs

    def __getitem__(self, index):
        rng = random.Random(f"{self.seed}:{self.epoch}:{index}")
        # Non-IID clients can have only one class, or no repeated labels.
        # Sample only feasible pair types; never retry indefinitely.
        if len(self.labels) == 1:
            similar = True
        elif not self.positive_labels:
            similar = False
        else:
            similar = index % 2 == 0
        if similar:
            label = rng.choice(self.positive_labels)
            first, second = rng.sample(self.indices[label], 2)
        else:
            first_label, second_label = rng.sample(self.labels, 2)
            first = rng.choice(self.indices[first_label])
            second = rng.choice(self.indices[second_label])
        return {
            "first": self.partition[first]["image"],
            "second": self.partition[second]["image"],
            "similar": int(similar),
        }


class SmellDataset:
    def __init__(self, partition, batch_size=64, validation_fraction=0.2, seed=42, num_workers=0):
        splits = partition.train_test_split(test_size=validation_fraction, seed=seed)
        self.train_partition = splits["train"]
        self.test_partition = splits["test"]
        self.batch_size = batch_size
        self.seed = seed
        self.num_workers = num_workers

    def dataloader(self, validation=False, shuffle=None, seed=None):
        partition = self.test_partition if validation else self.train_partition
        return DataLoader(
            partition.with_transform(transform_images),
            batch_size=self.batch_size,
            shuffle=not validation if shuffle is None else shuffle,
            num_workers=self.num_workers,
            worker_init_fn=seed_worker,
            generator=torch.Generator().manual_seed(self.seed if seed is None else seed),
        )

    def pair_loader(self, validation=False, pairs_per_batch=200, seed=None):
        partition = self.test_partition if validation else self.train_partition
        num_pairs = math.ceil(len(partition) / self.batch_size) * pairs_per_batch
        pairs = PairDataset(partition, num_pairs, self.seed if seed is None else seed)
        return DataLoader(
            pairs, batch_size=pairs_per_batch, num_workers=self.num_workers,
            worker_init_fn=seed_worker,
            generator=torch.Generator().manual_seed(pairs.seed),
        )
