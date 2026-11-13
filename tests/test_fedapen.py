"""FedAPEN regression tests with native Flower state and offline datasets."""

import unittest
from contextlib import redirect_stdout
import io
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import torch
from datasets import Dataset, DatasetDict, Features, Image as ImageFeature, Value
from flwr.common import Code, EvaluateRes, FitRes, RecordDict, Status, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server import History, Server
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import DirichletPartitioner
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader

from BaselineExperiment import BaselineExperiment
from FedAPEN import FedAPEN
from FedAPENDataset import FedAPENDataset, create_federated_dataset
from FedAPENExperiment import FedAPENExperiment, get_params
from Model import CIFARCNN, FashionMNISTCNN, create_model
from run import PlottingServer, create_apps, parse_args


class LocalDataset:
    def __init__(self, images, labels, batch_size=2):
        self.train_partition = [{"image": image, "label": label} for image, label in zip(images, labels)]
        self.adaptability_partition = self.train_partition
        self.test_partition = self.train_partition
        self.batch_size = batch_size

    def train_dataloader(self):
        return DataLoader(self.train_partition, batch_size=self.batch_size)

    def adaptability_dataloader(self):
        return DataLoader(self.adaptability_partition, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_partition, batch_size=self.batch_size)

    def dataloader(self, validation=False):
        return self.test_dataloader() if validation else self.train_dataloader()


class CountingLinear(nn.Linear):
    def __init__(self):
        super().__init__(2, 2, bias=False)
        self.forward_calls = 0

    def forward(self, images):
        self.forward_calls += 1
        return super().forward(images)


class ClientTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.images = torch.tensor([[1.0, 0.0], [0.0, 1.0], [2.0, 1.0]])
        self.labels = torch.tensor([0, 1, 0])
        self.dataset = LocalDataset(self.images, self.labels)

    def client(self, state=None, **kwargs):
        return FedAPENExperiment(
            CountingLinear(), CountingLinear(), self.dataset,
            state if state is not None else RecordDict(), device="cpu", **kwargs,
        )

    def test_private_state_survives_client_recreation_and_is_isolated(self):
        state = RecordDict()
        first = self.client(state)
        first.fit(first.get_parameters({}), {"epochs": 2, "server_round": 1})
        saved_private = get_params(first.private_model)
        with torch.no_grad():
            first.lambda_i.fill_(0.7)
        first.save_state()
        recreated = self.client(state)
        for saved, loaded in zip(saved_private, get_params(recreated.private_model)):
            np.testing.assert_array_equal(saved, loaded)
        self.assertAlmostEqual(recreated.lambda_i.item(), 0.7, places=6)
        other_client = self.client()
        self.assertEqual(other_client.lambda_i.item(), 0.5)
        self.assertFalse(np.array_equal(saved_private[0], get_params(other_client.private_model)[0]))

    def test_training_keeps_lambda_fixed_and_uses_one_forward_per_model_per_batch(self):
        client = self.client()
        parameters, count, _ = client.fit(client.get_parameters({}), {"epochs": 3})
        self.assertEqual(client.private_model.forward_calls, 6)
        self.assertEqual(client.shared_model.forward_calls, 6)
        self.assertEqual(client.lambda_i.item(), 0.5)
        self.assertIsNone(client.lambda_i.grad)
        self.assertEqual(count, 3)
        self.assertEqual(len(parameters), len(client.shared_model.state_dict()))

    def test_model_updates_match_detached_mutual_and_ensemble_gradients(self):
        images, labels = self.images[:2], self.labels[:2]
        dataset = LocalDataset(images, labels, batch_size=2)
        shared, private = nn.Linear(2, 2, bias=False), nn.Linear(2, 2, bias=False)
        initial_shared, initial_private = shared.weight.detach().clone(), private.weight.detach().clone()
        weight, lr = 0.25, 0.1
        with torch.no_grad():
            pv = torch.softmax(images @ initial_private.T, dim=1)
            pw = torch.softmax(images @ initial_shared.T, dim=1)
            pe = torch.softmax(weight * (images @ initial_private.T) + (1 - weight) * (images @ initial_shared.T), dim=1)
            target = torch.nn.functional.one_hot(labels, num_classes=2).float()
            private_gradient = ((pv - target) + (pv - pw) + weight * (pe - target)).T @ images / 2
            shared_gradient = ((pw - target) + (pw - pv) + (1 - weight) * (pe - target)).T @ images / 2
        client = FedAPENExperiment(
            shared, private, dataset, RecordDict(), learning_rate=lr,
            lambda_init=weight, epochs=1, device="cpu",
        )
        client.fit(get_params(shared), {})
        torch.testing.assert_close(private.weight, initial_private - lr * private_gradient)
        torch.testing.assert_close(shared.weight, initial_shared - lr * shared_gradient)

    def test_adaptation_changes_only_lambda_clips_and_persists_it(self):
        dataset = LocalDataset(torch.ones(3, 1), torch.zeros(3, dtype=torch.long))
        shared, private = nn.Linear(1, 2, bias=False), nn.Linear(1, 2, bias=False)
        with torch.no_grad():
            private.weight.copy_(torch.tensor([[3.0], [-3.0]]))
            shared.weight.copy_(torch.tensor([[-3.0], [3.0]]))
        state = RecordDict()
        client = FedAPENExperiment(shared, private, dataset, state, device="cpu")
        initial_shared, initial_private = get_params(shared), get_params(private)
        loss, count, metrics = client.evaluate(
            initial_shared, {"adaptability_epochs": 2, "adaptability_learning_rate": 100.0},
        )
        self.assertEqual(client.lambda_i.item(), 1.0)
        self.assertEqual(state["fedapen.personalization"]["lambda_i"], 1.0)
        self.assertEqual(metrics["accuracy"], 1.0)
        self.assertEqual(count, 3)
        self.assertTrue(np.isfinite(loss))
        for before, after in zip(initial_shared + initial_private, get_params(shared) + get_params(private)):
            np.testing.assert_array_equal(before, after)
        self.assertTrue(all(p.grad is None for p in list(shared.parameters()) + list(private.parameters())))

    def test_adam_and_learning_rate_are_used_by_both_experiments(self):
        client = self.client(optimizer="adam", learning_rate=0.02)
        client.fit(client.get_parameters({}), {"epochs": 1, "learning_rate": 0.003})
        for optimizer in (client.private_optimizer, client.shared_optimizer):
            self.assertIsInstance(optimizer, torch.optim.Adam)
            self.assertEqual(optimizer.param_groups[0]["lr"], 0.003)
        baseline = BaselineExperiment(nn.Linear(2, 2), self.dataset, optimizer="adam", momentum=0)
        baseline.fit(baseline.get_parameters({}), {"epochs": 1, "learning_rate": 0.004})
        self.assertIsInstance(baseline.optimizer, torch.optim.Adam)
        self.assertEqual(baseline.optimizer.param_groups[0]["lr"], 0.004)


class AggregationTests(unittest.TestCase):
    def test_shared_models_and_buffers_are_averaged_without_data_size_weights(self):
        strategy = FedAPEN(accept_failures=False)
        results = []
        for size, value in [(1, 2), (99, 6)]:
            arrays = [np.array([value], dtype=np.float32), np.array(value, dtype=np.int64)]
            result = FitRes(Status(Code.OK, ""), ndarrays_to_parameters(arrays), size, {"accuracy": value / 10})
            results.append((None, result))
        parameters, metrics = strategy.aggregate_fit(1, results, [])
        arrays = parameters_to_ndarrays(parameters)
        np.testing.assert_array_equal(arrays[0], np.array([4], dtype=np.float32))
        self.assertEqual(arrays[1].dtype, np.int64)
        self.assertEqual(arrays[1].item(), 4)
        self.assertAlmostEqual(metrics["accuracy"], 0.4)

    def test_mean_accuracy_and_bmta_are_client_means(self):
        strategy = FedAPEN(accept_failures=False)
        def results(accuracies):
            return [
                (None, EvaluateRes(Status(Code.OK, ""), 2.0, count, {"accuracy": accuracy}))
                for count, accuracy in zip((1, 99), accuracies)
            ]
        _, first = strategy.aggregate_evaluate(1, results((0.9, 0.1)), [])
        _, second = strategy.aggregate_evaluate(2, results((0.2, 0.4)), [])
        self.assertEqual(first["accuracy"], 0.5)
        self.assertAlmostEqual(second["accuracy"], 0.3)
        self.assertEqual(second["bmta"], 0.5)
        self.assertEqual(strategy.aggregate_evaluate(3, results((1.0, 1.0)), [RuntimeError()]), (None, {}))
        self.assertEqual(strategy.best_mean_accuracy, 0.5)


class DatasetAndModelTests(unittest.TestCase):
    def test_server_exports_charts_after_training_and_returns_original_history(self):
        history = History()
        history.add_metrics_distributed_fit(1, {"loss": 0.8})
        history.add_metrics_distributed(1, {"accuracy": 0.75})
        history.add_loss_distributed(1, 0.4)
        output = io.StringIO()
        with TemporaryDirectory() as directory:
            server = PlottingServer(FedAvg(), Path(directory))
            with patch.object(Server, "fit", return_value=(history, 1.25)) as fit:
                with redirect_stdout(output):
                    result, elapsed = server.fit(num_rounds=1, timeout=None)
            fit.assert_called_once_with(num_rounds=1, timeout=None)
            self.assertIs(result, history)
            self.assertEqual(elapsed, 1.25)
            self.assertEqual(
                {path.name for path in Path(directory).glob("*.png")},
                {"treino_loss.png", "avaliacao_accuracy.png", "avaliacao_loss.png"},
            )
            self.assertIn(str(Path(directory).resolve()), output.getvalue())

    def test_official_splits_are_merged_only_for_fedapen(self):
        raw = DatasetDict({
            "train": Dataset.from_dict({"label": [index % 10 for index in range(100)], "id": list(range(100))}),
            "test": Dataset.from_dict({"label": [index % 10 for index in range(20)], "id": list(range(100, 120))}),
        })
        with patch("flwr_datasets.federated_dataset.datasets.load_dataset", return_value=raw):
            fedapen = create_federated_dataset("offline", 2, 0.3, 42)
            fedapen_ids = set().union(*(set(fedapen.load_partition(cid)["id"]) for cid in range(2)))
            baseline = FederatedDataset(
                dataset="offline", seed=42,
                partitioners={"train": DirichletPartitioner(num_partitions=2, partition_by="label", alpha=0.3, seed=42)},
            )
            baseline_ids = set().union(*(set(baseline.load_partition(cid)["id"]) for cid in range(2)))
        self.assertEqual(fedapen_ids, set(range(120)))
        self.assertEqual(baseline_ids, set(range(100)))

    def test_apps_disable_central_evaluation_and_forward_cli_configuration(self):
        class LocalPartitions:
            def load_partition(self, cid):
                return Dataset.from_dict({"label": [0] * 30})

            def load_split(self, split):
                raise AssertionError("Central evaluation must not load the official test split.")

        for strategy in ("fedavg", "fedprox", "fedapen"):
            args = parse_args([
                "--strategy", strategy, "--num-clients", "4", "--clients-per-round", "2",
                "--fraction-evaluate", "0.5", "--learning-rate", "0.02", "--lr-decay", "0.8",
                "--local-epochs", "3", "--momentum", "0.2", "--num-rounds", "7",
            ])
            _, server_app = create_apps(args, LocalPartitions())
            components = server_app._server_fn(None)
            strategy = components.server.strategy
            self.assertIsNone(strategy.evaluate_fn)
            self.assertEqual(components.config.num_rounds, 7)
            self.assertEqual(strategy.fraction_evaluate, 0.5)
            self.assertEqual(strategy.min_fit_clients, 2)
            config = strategy.on_fit_config_fn(2)
            self.assertEqual(config["epochs"], 3)
            self.assertEqual(config["momentum"], 0.2)
            self.assertAlmostEqual(config["learning_rate"], 0.016)

    def test_three_partitions_are_disjoint_complete_and_reproducible(self):
        partition = Dataset.from_dict(
            {"image": [Image.fromarray(np.zeros((28, 28), dtype=np.uint8)) for _ in range(120)],
             "label": [index % 10 for index in range(120)], "id": list(range(120))},
            features=Features({"image": ImageFeature(), "label": Value("int64"), "id": Value("int64")}),
        )
        first, second = FedAPENDataset(partition), FedAPENDataset(partition)
        sets = [set(subset["id"]) for subset in (first.train_partition, first.adaptability_partition, first.test_partition)]
        self.assertEqual([len(subset) for subset in sets], [95, 5, 20])
        self.assertEqual(set.union(*sets), set(range(120)))
        self.assertFalse(sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
        self.assertEqual(first.adaptability_partition["id"], second.adaptability_partition["id"])
        batch = next(iter(first.adaptability_dataloader()))
        self.assertEqual(tuple(batch["image"].shape[1:]), (1, 28, 28))

    def test_empty_adaptability_partition_fails_clearly(self):
        partition = Dataset.from_dict({"image": [0] * 10, "label": [0] * 10})
        with self.assertRaisesRegex(ValueError, "nonempty"):
            FedAPENDataset(partition)

    def test_model_factory_and_initialization(self):
        for dataset, model_type, shape in [
            ("fashion-mnist", FashionMNISTCNN, (2, 1, 28, 28)),
            ("cifar10", CIFARCNN, (2, 3, 32, 32)),
        ]:
            model = create_model(dataset)
            self.assertIsInstance(model, model_type)
            self.assertEqual(tuple(model(torch.zeros(shape)).shape), (2, 10))
        torch.manual_seed(3)
        default = create_model("fashion-mnist", "none")
        torch.manual_seed(3)
        kaiming = create_model("fashion-mnist", "kaiming")
        self.assertFalse(torch.equal(default.conv1.weight, kaiming.conv1.weight))


if __name__ == "__main__":
    unittest.main()
