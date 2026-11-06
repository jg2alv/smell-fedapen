"""Exercise the real Flower/Ray apps with synthetic images and no downloads.

Run from the project root: python tests/smoke_simulation.py --strategy fedprox
"""

import argparse
from pathlib import Path
import sys

import numpy as np
import torch
from datasets import ClassLabel, Dataset, Features, Image as ImageFeature
from flwr.simulation import run_simulation
from flwr_datasets.partitioner import DirichletPartitioner
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from run import create_apps, parse_args


class InMemoryFederatedDataset:
    def __init__(self):
        rng = np.random.default_rng(42)
        images = [Image.fromarray(rng.integers(0, 256, (28, 28), dtype=np.uint8)) for _ in range(60)]
        data = Dataset.from_dict(
            {"image": images, "label": [index % 10 for index in range(60)]},
            features=Features({"image": ImageFeature(), "label": ClassLabel(num_classes=10)}),
        )
        self.test = data.select(range(40, 60))
        self.partitioner = DirichletPartitioner(
            num_partitions=2, partition_by="label", alpha=0.3, seed=42
        )
        self.partitioner.dataset = data.select(range(40))

    def load_split(self, split):
        assert split == "test"
        return self.test

    def load_partition(self, cid):
        return self.partitioner.load_partition(cid)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strategy", choices=("fedavg", "fedprox"), default="fedprox")
    choice = parser.parse_args()
    args = parse_args([
        "--strategy", choice.strategy, "--num-clients", "2", "--clients-per-round", "2",
        "--num-rounds", "2", "--local-epochs", "2", "--batch-size", "7",
        "--fraction-evaluate", "1", "--no-augmentation",
    ])
    client_app, server_app, history = create_apps(args, InMemoryFederatedDataset())
    if torch.cuda.is_available():
        num_gpus = 1.0
    else:
        num_gpus = 0.0
    run_simulation(
        server_app=server_app, client_app=client_app, num_supernodes=args.num_clients,
        backend_config={
            "client_resources": {"num_cpus": 1, "num_gpus": num_gpus},
            "init_args": {
                "num_cpus": 2, "include_dashboard": False,
                "object_store_memory": 100_000_000,
            },
        },
    )
    assert [entry["round"] for entry in history] == [0, 1, 2], history
    assert all(np.isfinite(entry["loss"]) and 0 <= entry["accuracy"] <= 1 for entry in history)
    print(f"{choice.strategy}: 2 rounds completed with centralized and client evaluation")


if __name__ == "__main__":
    main()
