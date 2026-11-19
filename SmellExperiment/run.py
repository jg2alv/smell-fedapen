"""Federated SMELL: run with `python SmellExperiment/run.py`."""

from datetime import datetime
from pathlib import Path
import random

import numpy as np
import torch
from flwr.client import ClientApp
from flwr.common import Context, ndarrays_to_parameters
from flwr.server import Server, ServerApp, ServerAppComponents, ServerConfig, SimpleClientManager
from flwr.server.strategy import FedAvg
from flwr.simulation import run_simulation
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import DirichletPartitioner

from SmellExperiment import SmellExperiment, get_params
from SmellDataset import SmellDataset
from SmellModel import SmellModel
from plots import plot_history


DATASET = "zalando-datasets/fashion_mnist"
NUM_CLIENTS = 10
CLIENTS_PER_ROUND = 3
PRETRAIN_ROUNDS = 10
PROTOTYPE_ROUNDS = 5
JOINT_ROUNDS = 25
NUM_ROUNDS = PRETRAIN_ROUNDS + PROTOTYPE_ROUNDS + JOINT_ROUNDS
LOCAL_EPOCHS = 1
BATCH_SIZE = 64
EMBEDDING_SIZE = 64
PAIRS_PER_BATCH = 200
VALIDATION_FRACTION = 0.2
DIRICHLET_ALPHA = 0.1
SEED = 42
# Baseline's ../output/graficos, resolved from the project rather than the cwd.
PLOT_DIRECTORY = Path(__file__).resolve().parents[2] / "output" / "graficos"


class PlottingServer(Server):
    def __init__(self, strategy, plot_directory):
        super().__init__(client_manager=SimpleClientManager(), strategy=strategy)
        self.plot_directory = Path(plot_directory)

    def fit(self, num_rounds, timeout):
        history, elapsed = super().fit(num_rounds=num_rounds, timeout=timeout)
        paths = plot_history(history, self.plot_directory)
        print(f"{len(paths)} gráficos salvos em {self.plot_directory.resolve()}")
        return history, elapsed


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def round_config(server_round):
    if server_round <= PRETRAIN_ROUNDS:
        phase = "pretrain"
    elif server_round <= PRETRAIN_ROUNDS + PROTOTYPE_ROUNDS:
        phase = "prototypes"
    else:
        phase = "joint"
    return {"phase": phase, "server_round": server_round, "epochs": LOCAL_EPOCHS}


def weighted_average(results):
    total = sum(size for size, _ in results)
    return {
        name: sum(size * metrics[name] for size, metrics in results) / total
        for name in results[0][1]
    }


def create_apps(federated_dataset=None):
    seed_everything(SEED)
    if federated_dataset is None:
        partitioner = DirichletPartitioner(
            num_partitions=NUM_CLIENTS, partition_by="label", alpha=DIRICHLET_ALPHA, seed=SEED,
        )
        federated_dataset = FederatedDataset(
            dataset=DATASET, partitioners={"train": partitioner}, seed=SEED,
        )
    federated_dataset.load_partition(0)

    def client_fn(context: Context):
        cid = int(context.node_config["partition-id"])
        seed_everything(SEED + cid)
        torch.set_num_threads(1)
        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
        dataset = SmellDataset(
            federated_dataset.load_partition(cid), batch_size=BATCH_SIZE,
            validation_fraction=VALIDATION_FRACTION, seed=SEED + cid,
        )
        return SmellExperiment(
            SmellModel(EMBEDDING_SIZE), dataset, state=context.state,
            device=device, seed=SEED + cid, pairs_per_batch=PAIRS_PER_BATCH,
        ).to_client()

    def server_fn(context: Context):
        seed_everything(SEED)
        model = SmellModel(EMBEDDING_SIZE)
        strategy = FedAvg(
            fraction_fit=CLIENTS_PER_ROUND / NUM_CLIENTS,
            min_fit_clients=CLIENTS_PER_ROUND,
            min_available_clients=NUM_CLIENTS,
            fraction_evaluate=1.0,
            min_evaluate_clients=NUM_CLIENTS,
            initial_parameters=ndarrays_to_parameters(get_params(model)),
            on_fit_config_fn=round_config,
            on_evaluate_config_fn=round_config,
            fit_metrics_aggregation_fn=weighted_average,
            evaluate_metrics_aggregation_fn=weighted_average,
            accept_failures=False,
        )
        plot_directory = PLOT_DIRECTORY / f"fashion-mnist_smell_{datetime.now():%Y%m%d_%H%M%S_%f}"
        return ServerAppComponents(
            server=PlottingServer(strategy, plot_directory),
            config=ServerConfig(num_rounds=NUM_ROUNDS),
        )

    return ClientApp(client_fn=client_fn), ServerApp(server_fn=server_fn)


def main():
    client_app, server_app = create_apps()
    if torch.cuda.is_available():
        num_gpus = 1.0
    else:
        num_gpus = 0.0
    run_simulation(
        server_app=server_app, client_app=client_app, num_supernodes=NUM_CLIENTS,
        backend_config={
            "client_resources": {"num_cpus": 1, "num_gpus": num_gpus},
            "init_args": {"include_dashboard": False},
        },
    )


if __name__ == "__main__":
    main()
