"""Run local Flower/PyTorch experiments with Fashion-MNIST or CIFAR-10."""

import argparse
import random

import numpy as np
import torch
from flwr.client import ClientApp
from flwr.common import Context, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg, FedProx
from flwr.simulation import run_simulation
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import DirichletPartitioner

from BaselineDataset import BaselineDataset
from BaselineExperiment import BaselineExperiment, get_params
from Model import Model

# PARA IGNORAR OS WARNINGS DE FEATURE UPDATED E DEPRECATION DO FLWR
import logging

class IgnoreWarnings(logging.Filter):
    def filter(self, record):
        return record.levelno != logging.WARNING

logging.getLogger("flwr").addFilter(IgnoreWarnings())

DATASETS = {
    "fashion-mnist": ("zalando-datasets/fashion_mnist", 784),
    "cifar10": ("uoft-cs/cifar10", 3072),
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GPUS_PER_CLIENT = 1.0 if torch.cuda.is_available() else 0.0


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Simulação federada de Fashion-MNIST ou CIFAR-10 com PyTorch e Flower.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--strategy", choices=("fedavg", "fedprox"), default="fedprox", help="estratégia do Flower")
    parser.add_argument("--num-clients", type=int, default=100, help="total de clientes simulados")
    parser.add_argument("--clients-per-round", type=int, default=3, help="clientes que treinam em cada rodada")
    parser.add_argument("--num-rounds", type=int, default=100, help="número de rodadas federadas")
    parser.add_argument("--mu", type=float, default=0.01, help="coeficiente proximal do FedProx")
    parser.add_argument("--fraction-evaluate", type=float, default=0.3, help="fração de clientes para validação local")
    parser.add_argument("--local-epochs", type=int, default=1, help="épocas locais por rodada")
    parser.add_argument("--batch-size", type=int, default=32, help="tamanho do lote")
    parser.add_argument("--learning-rate", type=float, default=0.01, help="taxa de aprendizado do SGD")
    parser.add_argument("--momentum", type=float, default=0.9, help="momentum do SGD")
    parser.add_argument("--seed", type=int, default=42, help="seed dos geradores aleatórios")
    parser.add_argument("--no-augmentation", dest="augment", action="store_false", help="desativa recorte e espelhamento aleatórios")
    parser.add_argument("--dataset", choices=DATASETS, default="fashion-mnist", help="conjunto de imagens")
    parser.add_argument("--alpha", type=float, default=0.3, help="concentração da partição Dirichlet")
    parser.add_argument("--validation-fraction", type=float, default=0.2, help="fração local reservada à validação")
    parser.add_argument("--num-workers", type=int, default=0, help="workers de cada DataLoader")
    return parser.parse_args(argv)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def weighted_average(results):
    """Aggregate scalar client metrics using their number of examples."""
    total_examples = sum(num_examples for num_examples, _ in results)
    keys = set.union(*(set(metrics) for _, metrics in results))
    return {
        key: sum(num_examples * float(metrics[key]) for num_examples, metrics in results)
        / total_examples
        for key in sorted(keys)
    }


def create_apps(args, federated_dataset=None):
    """Build native Flower apps; dataset injection also permits offline tests."""
    dataset_path, input_size = DATASETS[args.dataset]
    seed_everything(args.seed)
    torch.set_num_threads(1)
    if federated_dataset is None:
        partitioner = DirichletPartitioner(
            num_partitions=args.num_clients, partition_by="label",
            alpha=args.alpha, seed=args.seed,
        )
        federated_dataset = FederatedDataset(
            dataset=dataset_path, partitioners={"train": partitioner}, seed=args.seed,
        )
    # Load once before Ray workers start, avoiding simultaneous downloads.
    test_partition = federated_dataset.load_split("test")
    federated_dataset.load_partition(0)
    history = []

    def client_fn(context: Context):
        cid = int(context.node_config["partition-id"])
        client_seed = args.seed + cid
        seed_everything(client_seed)
        torch.set_num_threads(1)
        dataset = BaselineDataset(
            federated_dataset.load_partition(cid), batch_size=args.batch_size,
            validation_fraction=args.validation_fraction, seed=client_seed,
            num_workers=args.num_workers, augment=args.augment,
            dataset_name=args.dataset,
        )
        return BaselineExperiment(
            Model(input_size=input_size), dataset, learning_rate=args.learning_rate,
            momentum=args.momentum, epochs=args.local_epochs,
            device=DEVICE, seed=client_seed,
        ).to_client()

    def server_fn(context: Context):
        seed_everything(args.seed)
        model = Model(input_size=input_size)
        initial_parameters = ndarrays_to_parameters(get_params(model))
        dataset = BaselineDataset.from_test_partition(
            test_partition, batch_size=args.batch_size,
            num_workers=args.num_workers, seed=args.seed,
            dataset_name=args.dataset,
        )
        evaluator = BaselineExperiment(model, dataset, device=DEVICE, seed=args.seed)

        def evaluate(server_round, parameters, config):
            loss, _, metrics = evaluator.evaluate(parameters, config)
            history.append({"round": server_round, **metrics})
            return loss, metrics

        strategy_kwargs = dict(
            fraction_fit=args.clients_per_round / args.num_clients,
            min_fit_clients=args.clients_per_round,
            min_available_clients=args.num_clients,
            fraction_evaluate=args.fraction_evaluate,
            min_evaluate_clients=1 if args.fraction_evaluate > 0 else 0,
            initial_parameters=initial_parameters,
            evaluate_fn=evaluate,
            on_fit_config_fn=lambda server_round: {
                "server_round": server_round,
                "epochs": args.local_epochs,
                "learning_rate": args.learning_rate,
                "momentum": args.momentum,
            },
            fit_metrics_aggregation_fn=weighted_average,
            evaluate_metrics_aggregation_fn=weighted_average,
            accept_failures=False,
        )
        if args.strategy == "fedprox":
            strategy = FedProx(proximal_mu=args.mu, **strategy_kwargs)
        else:
            strategy = FedAvg(**strategy_kwargs)
        return ServerAppComponents(
            strategy=strategy, config=ServerConfig(num_rounds=args.num_rounds),
        )

    return ClientApp(client_fn=client_fn), ServerApp(server_fn=server_fn), history


def main(argv=None):
    args = parse_args(argv)
    client_app, server_app, history = create_apps(args)
    run_simulation(
        server_app=server_app, client_app=client_app,
        num_supernodes=args.num_clients,
        backend_config={
            "client_resources": {
                "num_cpus": 1,
                "num_gpus": GPUS_PER_CLIENT,
            },
            "init_args": {"include_dashboard": False},
        },
    )
    return history


if __name__ == "__main__":
    main()
