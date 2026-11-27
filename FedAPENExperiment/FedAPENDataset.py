"""Local train/adaptability/test partitions for the FedAPEN protocol."""

import random

import numpy as np
import torch
from datasets import Dataset
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import DirichletPartitioner
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, RandomCrop, RandomHorizontalFlip, ToTensor


NORMALIZATION = {
    "fashion-mnist": ((0.2860,), (0.3205,)),
    "cifar10": ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
}
IMAGE_SIZE = {"fashion-mnist": 28, "cifar10": 32}


def create_federated_dataset(dataset_path, num_clients, alpha, seed):
    """Merge the official splits before partitioning, as in the reference code."""
    return FederatedDataset(
        dataset=dataset_path,
        preprocessor={"train": ("train", "test")},
        partitioners={
            "train": DirichletPartitioner(
                num_partitions=num_clients, partition_by="label", alpha=alpha, seed=seed,
            ),
        },
        seed=seed,
    )


def seed_worker(worker_id):
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)


class FedAPENDataset:
    def __init__(
        self, partition: Dataset, *, dataset_name="fashion-mnist", batch_size=64,
        validation_fraction=1 / 6, adaptability_fraction=0.05, seed=42,
        num_workers=0, augment=True,
    ):
        if not 0 < adaptability_fraction < 1:
            raise ValueError("adaptability_fraction must be between zero and one.")
        self.batch_size = batch_size
        self.num_workers = num_workers
        if dataset_name == "cifar10" and "image" not in partition.column_names:
            partition = partition.rename_column("img", "image")
        splits = partition.train_test_split(test_size=validation_fraction, seed=seed)
        train_pool = splits["train"]
        adaptability_size = int(len(train_pool) * adaptability_fraction)
        if not 1 <= adaptability_size < len(train_pool):
            raise ValueError("The partition must contain nonempty training and adaptability sets.")
        subsets = train_pool.train_test_split(test_size=adaptability_size, seed=seed)
        self.train_partition = subsets["train"]
        self.adaptability_partition = subsets["test"]
        self.test_partition = splits["test"]

        self.eval_transforms = Compose([ToTensor(), Normalize(*NORMALIZATION[dataset_name])])
        self.train_transforms = self.eval_transforms
        if augment:
            self.train_transforms = Compose([
                RandomCrop(IMAGE_SIZE[dataset_name], padding=4), RandomHorizontalFlip(),
                ToTensor(), Normalize(*NORMALIZATION[dataset_name]),
            ])
        self._train_generator = torch.Generator().manual_seed(seed)
        self._adaptability_generator = torch.Generator().manual_seed(seed + 1)
        self._test_generator = torch.Generator().manual_seed(seed + 2)

    def apply_train_transforms(self, batch):
        batch["image"] = [self.train_transforms(image) for image in batch["image"]]
        return batch

    def apply_eval_transforms(self, batch):
        batch["image"] = [self.eval_transforms(image) for image in batch["image"]]
        return batch

    def train_dataloader(self):
        return DataLoader(
            self.train_partition.with_transform(self.apply_train_transforms),
            batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers,
            worker_init_fn=seed_worker, generator=self._train_generator,
        )

    def adaptability_dataloader(self):
        return DataLoader(
            self.adaptability_partition.with_transform(self.apply_eval_transforms),
            batch_size=min(self.batch_size, len(self.adaptability_partition)),
            shuffle=True, num_workers=self.num_workers,
            worker_init_fn=seed_worker, generator=self._adaptability_generator,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_partition.with_transform(self.apply_eval_transforms),
            batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers,
            worker_init_fn=seed_worker, generator=self._test_generator,
        )
