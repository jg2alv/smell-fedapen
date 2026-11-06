"""Run local Flower/PyTorch experiments with Fashion-MNIST or CIFAR-10."""

import argparse
import logging
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
from FedAPEN import FedAPEN
from FedAPENDataset import FedAPENDataset, create_federated_dataset
from FedAPENExperiment import FedAPENExperiment
from Model import create_model

# PARA IGNORAR OS WARNINGS DE FEATURE UPDATED E DEPRECATION DO FLWR
class IgnoreWarnings(logging.Filter):
    def filter(self, record):
        return record.levelno != logging.WARNING

logging.getLogger("flwr").addFilter(IgnoreWarnings())

DATASETS = {
    "fashion-mnist": "zalando-datasets/fashion_mnist",
    "cifar10": "uoft-cs/cifar10",
}

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    GPUS_PER_CLIENT = 1.0
else:
    DEVICE = torch.device("cpu")
    GPUS_PER_CLIENT = 0.0


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Simulação federada de Fashion-MNIST ou CIFAR-10 com PyTorch e Flower.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--strategy", choices=("fedavg", "fedprox", "fedapen"), default="fedprox", help="estratégia federada")
    parser.add_argument("--num-clients", type=int, default=10, help="total de clientes simulados")
    parser.add_argument("--clients-per-round", type=int, default=10, help="clientes que treinam em cada rodada")
    parser.add_argument("--num-rounds", type=int, default=100, help="número de rodadas federadas")
    parser.add_argument("--mu", type=float, default=0.001, help="coeficiente proximal do FedProx")
    parser.add_argument("--fraction-evaluate", type=float, default=1.0, help="fração de clientes para avaliação local")
    parser.add_argument("--local-epochs", type=int, default=5, help="épocas locais por rodada")
    parser.add_argument("--batch-size", type=int, default=64, help="tamanho do lote")
    parser.add_argument("--learning-rate", type=float, default=0.01, help="taxa de aprendizado do SGD")
    parser.add_argument("--momentum", type=float, default=0.0, help="momentum do SGD")
    parser.add_argument("--optimizer", choices=("sgd", "adam"), default="sgd", help="otimizador dos modelos")
    parser.add_argument("--init", choices=("kaiming", "xavier", "none"), default="kaiming", help="inicialização dos pesos")
    parser.add_argument("--seed", type=int, default=42, help="seed dos geradores aleatórios")
    parser.add_argument("--no-augmentation", dest="augment", action="store_false", help="desativa recorte e espelhamento aleatórios")
    parser.add_argument("--dataset", choices=DATASETS, default="fashion-mnist", help="conjunto de imagens")
    parser.add_argument("--alpha", type=float, default=0.1, help="concentração da partição Dirichlet")
    parser.add_argument("--validation-fraction", type=float, default=1 / 6, help="fração local reservada ao teste")
    parser.add_argument("--num-workers", type=int, default=0, help="workers de cada DataLoader")
    parser.add_argument("--adaptability-fraction", type=float, default=0.05, help="fração do treino reservada à adaptação do FedAPEN")
    parser.add_argument("--adaptability-epochs", type=int, default=10, help="épocas de aprendizado de lambda no FedAPEN")
    parser.add_argument("--adaptability-learning-rate", type=float, default=0.001, help="learning rate de lambda no FedAPEN")
    parser.add_argument("--lambda-init", type=float, default=0.5, help="peso privado inicial do FedAPEN")
    parser.add_argument("--lr-decay", type=float, default=0.99, help="fator de decaimento do learning rate por rodada")
    args = parser.parse_args(argv)
    if not 1 <= args.clients_per_round <= args.num_clients:
        parser.error("--clients-per-round deve estar entre 1 e --num-clients.")
    return args


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
    dataset_path = DATASETS[args.dataset]
    seed_everything(args.seed)
    torch.set_num_threads(1)
    if federated_dataset is None:
        if args.strategy == "fedapen":
            federated_dataset = create_federated_dataset(
                dataset_path, args.num_clients, args.alpha, args.seed,
            )
        else:
            partitioner = DirichletPartitioner(
                num_partitions=args.num_clients, partition_by="label",
                alpha=args.alpha, seed=args.seed,
            )
            federated_dataset = FederatedDataset(
                dataset=dataset_path, partitioners={"train": partitioner}, seed=args.seed,
            )
    # Load once before Ray workers start, avoiding simultaneous downloads.
    federated_dataset.load_partition(0)

    def client_fn(context: Context):
        cid = int(context.node_config["partition-id"])
        client_seed = args.seed + cid
        seed_everything(client_seed)
        torch.set_num_threads(1)
        if args.strategy == "fedapen":
            dataset = FedAPENDataset(
                federated_dataset.load_partition(cid), dataset_name=args.dataset,
                batch_size=args.batch_size, validation_fraction=args.validation_fraction,
                adaptability_fraction=args.adaptability_fraction, seed=client_seed,
                num_workers=args.num_workers, augment=args.augment,
            )
            return FedAPENExperiment(
                create_model(args.dataset, args.init), create_model(args.dataset, args.init),
                dataset, context.state,
                learning_rate=args.learning_rate, momentum=args.momentum,
                optimizer=args.optimizer,
                epochs=args.local_epochs, adaptability_epochs=args.adaptability_epochs,
                adaptability_learning_rate=args.adaptability_learning_rate,
                lambda_init=args.lambda_init, device=DEVICE, seed=client_seed,
            ).to_client()
        dataset = BaselineDataset(
            federated_dataset.load_partition(cid), batch_size=args.batch_size,
            validation_fraction=args.validation_fraction, seed=client_seed,
            num_workers=args.num_workers, augment=args.augment,
            dataset_name=args.dataset,
        )
        return BaselineExperiment(
            create_model(args.dataset, args.init), dataset, learning_rate=args.learning_rate,
            momentum=args.momentum, optimizer=args.optimizer, epochs=args.local_epochs,
            device=DEVICE, seed=client_seed,
        ).to_client()

    def server_fn(context: Context):
        seed_everything(args.seed)
        model = create_model(args.dataset, args.init)
        initial_parameters = ndarrays_to_parameters(get_params(model))

        strategy_kwargs = dict(
            fraction_fit=args.clients_per_round / args.num_clients,
            min_fit_clients=args.clients_per_round,
            min_available_clients=args.num_clients,
            fraction_evaluate=args.fraction_evaluate,
            min_evaluate_clients=1 if args.fraction_evaluate > 0 else 0,
            initial_parameters=initial_parameters,
            on_fit_config_fn=lambda server_round: {
                "server_round": server_round,
                "epochs": args.local_epochs,
                "learning_rate": args.learning_rate * args.lr_decay ** (server_round - 1),
                "momentum": args.momentum,
            },
            accept_failures=False,
        )
        if args.strategy == "fedapen":
            strategy = FedAPEN(
                on_evaluate_config_fn=lambda server_round: {
                    "server_round": server_round,
                    "adaptability_epochs": args.adaptability_epochs,
                    "adaptability_learning_rate": args.adaptability_learning_rate,
                },
                **strategy_kwargs,
            )
        else:
            strategy_kwargs.update(
                fit_metrics_aggregation_fn=weighted_average,
                evaluate_metrics_aggregation_fn=weighted_average,
            )
            if args.strategy == "fedprox":
                strategy = FedProx(proximal_mu=args.mu, **strategy_kwargs)
            else:
                strategy = FedAvg(**strategy_kwargs)
        return ServerAppComponents(
            strategy=strategy, config=ServerConfig(num_rounds=args.num_rounds),
        )

    return ClientApp(client_fn=client_fn), ServerApp(server_fn=server_fn)


def main(argv=None):
    args = parse_args(argv)
    client_app, server_app = create_apps(args)
    return run_simulation(
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


if __name__ == "__main__":
    main()
