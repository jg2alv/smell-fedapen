"""Run local Flower/PyTorch experiments with Fashion-MNIST or CIFAR-10."""

import argparse
import math
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

from FMNISTDataset import FMNISTDataset
from PoCExperimentFMNIST import PoCExperimentFMNIST, get_params
from PoCModel_2HiddenLayers import PoCModel_2HiddenLayers

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


def positive_int(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("deve ser um inteiro positivo")
    return value


def nonnegative_int(value):
    value = int(value)
    if value < 0:
        raise argparse.ArgumentTypeError("deve ser um inteiro não negativo")
    return value


def finite_float(value):
    value = float(value)
    if not math.isfinite(value):
        raise argparse.ArgumentTypeError("deve ser um número finito")
    return value


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Simulação federada de Fashion-MNIST ou CIFAR-10 com PyTorch e Flower.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    federation = parser.add_argument_group("federação")
    federation.add_argument("--strategy", choices=("fedavg", "fedprox"), default="fedprox", help="estratégia do Flower")
    federation.add_argument("--num-clients", type=positive_int, default=100, help="total de clientes simulados")
    federation.add_argument("--clients-per-round", type=positive_int, default=3, help="clientes que treinam em cada rodada")
    federation.add_argument("--num-rounds", type=positive_int, default=100, help="número de rodadas federadas")
    federation.add_argument("--mu", type=finite_float, default=0.01, help="coeficiente proximal do FedProx")
    federation.add_argument(
        "--fraction-evaluate", type=finite_float, default=0.3,
        help="fração de clientes para validação local; 0 mantém apenas o teste central",
    )
    training = parser.add_argument_group("treinamento")
    training.add_argument("--local-epochs", type=positive_int, default=1, help="épocas locais por rodada")
    training.add_argument("--batch-size", type=positive_int, default=32, help="tamanho do lote")
    training.add_argument("--learning-rate", type=finite_float, default=0.01, help="taxa de aprendizado do SGD")
    training.add_argument("--momentum", type=finite_float, default=0.9, help="momentum do SGD")
    training.add_argument("--seed", type=nonnegative_int, default=42, help="seed dos geradores aleatórios")
    training.add_argument("--no-augmentation", dest="augment", action="store_false", help="desativa recorte e espelhamento aleatórios")
    data = parser.add_argument_group("dados")
    data.add_argument("--dataset", choices=DATASETS, default="fashion-mnist", help="conjunto de imagens")
    data.add_argument("--alpha", type=finite_float, default=0.3, help="concentração da partição Dirichlet")
    data.add_argument("--validation-fraction", type=finite_float, default=0.2, help="fração local reservada à validação")
    data.add_argument("--num-workers", type=nonnegative_int, default=0, help="workers de cada DataLoader")
    args = parser.parse_args(argv)

    if args.clients_per_round > args.num_clients:
        parser.error("--clients-per-round não pode exceder --num-clients")
    if args.mu < 0:
        parser.error("--mu deve ser não negativo")
    if args.learning_rate <= 0:
        parser.error("--learning-rate deve ser positivo")
    if not 0 <= args.momentum < 1:
        parser.error("--momentum deve pertencer a [0, 1)")
    if args.alpha <= 0:
        parser.error("--alpha deve ser positivo")
    if not 0 < args.validation_fraction < 1:
        parser.error("--validation-fraction deve pertencer a (0, 1)")
    if not 0 <= args.fraction_evaluate <= 1:
        parser.error("--fraction-evaluate deve pertencer a [0, 1]")
    if args.seed >= 2**32:
        parser.error("--seed deve ser menor que 2**32")
    return args


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def weighted_average(results):
    """Aggregate scalar client metrics using their number of examples."""
    total_examples = sum(num_examples for num_examples, _ in results)
    if total_examples == 0:
        return {}
    keys = set.intersection(*(set(metrics) for _, metrics in results))
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
        dataset = FMNISTDataset(
            federated_dataset.load_partition(cid), batch_size=args.batch_size,
            validation_fraction=args.validation_fraction, seed=client_seed,
            num_workers=args.num_workers, augment=args.augment,
            dataset_name=args.dataset,
        )
        return PoCExperimentFMNIST(
            PoCModel_2HiddenLayers(input_size=input_size), dataset, learning_rate=args.learning_rate,
            momentum=args.momentum, epochs=args.local_epochs,
            device=DEVICE, seed=client_seed,
        ).to_client()

    def server_fn(context: Context):
        seed_everything(args.seed)
        model = PoCModel_2HiddenLayers(input_size=input_size)
        initial_parameters = ndarrays_to_parameters(get_params(model))
        dataset = FMNISTDataset.from_test_partition(
            test_partition, batch_size=args.batch_size,
            num_workers=args.num_workers, seed=args.seed,
            dataset_name=args.dataset,
        )
        evaluator = PoCExperimentFMNIST(model, dataset, device=DEVICE, seed=args.seed)

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
    if len(history) != args.num_rounds + 1:
        raise RuntimeError("A simulação terminou sem avaliar todas as rodadas.")
    return history


if __name__ == "__main__":
    main()
