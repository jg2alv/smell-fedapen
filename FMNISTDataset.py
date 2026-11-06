"""Native PyTorch data loaders for Fashion-MNIST and CIFAR-10 partitions."""

import random

import numpy as np
import torch
from datasets import Dataset
from torch.utils.data import DataLoader
from torchvision.transforms import (
    Compose,
    Normalize,
    RandomCrop,
    RandomHorizontalFlip,
    ToTensor,
)


# Preserve the preprocessing used by the original experiment.
FM_NORMALIZATION = ((0.1307,), (0.3081,))
EVAL_TRANSFORMS = Compose([ToTensor(), Normalize(*FM_NORMALIZATION)])
TRAIN_TRANSFORMS = Compose(
    [
        RandomCrop(28, padding=4),
        RandomHorizontalFlip(),
        ToTensor(),
        Normalize(*FM_NORMALIZATION),
    ]
)

CIFAR_NORMALIZATION = ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
CIFAR_EVAL_TRANSFORMS = Compose([ToTensor(), Normalize(*CIFAR_NORMALIZATION)])
CIFAR_TRAIN_TRANSFORMS = Compose(
    [
        RandomCrop(32, padding=4),
        RandomHorizontalFlip(),
        ToTensor(),
        Normalize(*CIFAR_NORMALIZATION),
    ]
)


def seed_worker(worker_id: int) -> None:
    """Seed libraries using the worker seed assigned by PyTorch."""
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


class FMNISTDataset:
    """Prepare one client's train/validation split without a platform API."""

    def __init__(
        self,
        partition: Dataset,
        *,
        batch_size: int = 32,
        validation_fraction: float = 0.2,
        seed: int = 42,
        num_workers: int = 0,
        shuffle: bool = True,
        augment: bool = True,
        dataset_name: str = "fashion-mnist",
    ) -> None:
        self._configure(batch_size, seed, num_workers, shuffle, augment, dataset_name)
        partition = self._prepare_partition(partition)
        splits = partition.train_test_split(
            test_size=validation_fraction, seed=seed
        )
        self.train_partition = splits["train"]
        self.test_partition = splits["test"]

    def _configure(
        self,
        batch_size: int,
        seed: int,
        num_workers: int,
        shuffle: bool,
        augment: bool,
        dataset_name: str,
    ) -> None:
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.seed = seed
        self.dataset_name = dataset_name
        train_transforms, self.eval_transforms = {
            "fashion-mnist": (TRAIN_TRANSFORMS, EVAL_TRANSFORMS),
            "cifar10": (CIFAR_TRAIN_TRANSFORMS, CIFAR_EVAL_TRANSFORMS),
        }[dataset_name]
        self.train_transforms = train_transforms if augment else self.eval_transforms
        self._train_generator = torch.Generator().manual_seed(seed)
        self._eval_generator = torch.Generator().manual_seed(seed)

    def _prepare_partition(self, partition: Dataset) -> Dataset:
        if self.dataset_name == "cifar10" and "image" not in partition.column_names:
            return partition.rename_column("img", "image")
        return partition

    @classmethod
    def from_test_partition(
        cls,
        partition: Dataset,
        *,
        batch_size: int = 32,
        num_workers: int = 0,
        seed: int = 42,
        dataset_name: str = "fashion-mnist",
    ) -> "FMNISTDataset":
        """Use the complete official test split for centralized evaluation."""
        instance = cls.__new__(cls)
        instance._configure(batch_size, seed, num_workers, False, False, dataset_name)
        instance.train_partition = None
        instance.test_partition = instance._prepare_partition(partition)
        return instance

    def dataloader(self, validation: bool = False) -> DataLoader:
        partition = self.validation() if validation else self.train()
        return DataLoader(
            partition,
            batch_size=self.batch_size,
            shuffle=self.shuffle if not validation else False,
            num_workers=self.num_workers,
            worker_init_fn=seed_worker,
            generator=self._eval_generator if validation else self._train_generator,
        )

    def apply_train_transforms(self, batch: dict) -> dict:
        batch["image"] = [self.train_transforms(image) for image in batch["image"]]
        return batch

    def apply_eval_transforms(self, batch: dict) -> dict:
        batch["image"] = [self.eval_transforms(image) for image in batch["image"]]
        return batch

    def train(self) -> Dataset:
        return self.train_partition.with_transform(self.apply_train_transforms)

    def validation(self) -> Dataset:
        return self.test_partition.with_transform(self.apply_eval_transforms)
