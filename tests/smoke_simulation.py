"""Exercise the real Flower/Ray apps with synthetic images and no downloads.

Run from the project root: python tests/smoke_simulation.py --strategy fedprox
"""

import argparse
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
import torch
from datasets import ClassLabel, Dataset, Features, Image as ImageFeature
from flwr.simulation import run_simulation
from flwr.server.strategy import FedAvg, FedProx
from flwr_datasets.partitioner import DirichletPartitioner
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from run import create_apps, parse_args
from FedAPEN import FedAPEN


class InMemoryFederatedDataset:
    def __init__(self, dataset_name, merge_splits):
        rng = np.random.default_rng(42)
        shape = (32, 32, 3) if dataset_name == "cifar10" else (28, 28)
        images = [Image.fromarray(rng.integers(0, 256, shape, dtype=np.uint8)) for _ in range(120)]
        data = Dataset.from_dict(
            {"image": images, "label": [index % 10 for index in range(120)]},
            features=Features({"image": ImageFeature(), "label": ClassLabel(num_classes=10)}),
        )
        self.partitioner = DirichletPartitioner(
            num_partitions=2, partition_by="label", alpha=0.3, seed=42, min_partition_size=30,
        )
        self.partitioner.dataset = data if merge_splits else data.select(range(100))

    def load_partition(self, cid):
        return self.partitioner.load_partition(cid)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strategy", choices=("fedavg", "fedprox", "fedapen"), default="fedapen")
    parser.add_argument("--dataset", choices=("fashion-mnist", "cifar10"), default="fashion-mnist")
    choice = parser.parse_args()
    args = parse_args([
        "--strategy", choice.strategy, "--dataset", choice.dataset,
        "--num-clients", "2", "--clients-per-round", "2",
        "--num-rounds", "2", "--local-epochs", "2", "--batch-size", "7",
        "--fraction-evaluate", "1", "--no-augmentation",
    ])
    client_app, server_app = create_apps(
        args, InMemoryFederatedDataset(choice.dataset, choice.strategy == "fedapen"),
    )
    if torch.cuda.is_available():
        num_gpus = 1.0
    else:
        num_gpus = 0.0
    strategy_type = {"fedavg": FedAvg, "fedprox": FedProx, "fedapen": FedAPEN}[choice.strategy]
    aggregate_fit = strategy_type.aggregate_fit
    aggregate_evaluate = strategy_type.aggregate_evaluate
    fit_rounds, evaluate_rounds = [], []

    def record_fit(strategy, server_round, results, failures):
        fit_rounds.append((server_round, len(results), len(failures)))
        return aggregate_fit(strategy, server_round, results, failures)

    def record_evaluate(strategy, server_round, results, failures):
        evaluate_rounds.append((server_round, len(results), len(failures)))
        return aggregate_evaluate(strategy, server_round, results, failures)

    with patch.object(strategy_type, "aggregate_fit", record_fit), patch.object(strategy_type, "aggregate_evaluate", record_evaluate):
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
    assert fit_rounds == [(1, 2, 0), (2, 2, 0)], fit_rounds
    assert evaluate_rounds == [(1, 2, 0), (2, 2, 0)], evaluate_rounds
    print(f"{choice.strategy}/{choice.dataset}: 2 rounds completed with local evaluation")


if __name__ == "__main__":
    main()
