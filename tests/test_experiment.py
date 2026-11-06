"""Regression tests using small, deterministic datasets; no downloads required."""

import unittest

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from BaselineExperiment import BaselineExperiment, get_params, set_params


class DictDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return {"image": self.images[index], "label": self.labels[index]}


class LocalDataset:
    def __init__(self, images, labels, batch_size=2):
        self.train_partition = DictDataset(images, labels)
        self.test_partition = self.train_partition
        self.batch_size = batch_size

    def dataloader(self, validation=False):
        partition = self.test_partition if validation else self.train_partition
        return DataLoader(partition, batch_size=self.batch_size, shuffle=False)


class RecordingLinear(nn.Linear):
    def __init__(self, in_features=2, out_features=2):
        super().__init__(in_features, out_features, bias=False)
        self.outputs = []

    def forward(self, images):
        outputs = super().forward(images)
        self.outputs.append(outputs.detach().clone())
        return outputs


class ExperimentTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.images = torch.tensor(
            [[1.0, 0.0], [0.0, 1.0], [2.0, -1.0], [-1.0, 2.0], [3.0, 1.0]]
        )
        self.labels = torch.tensor([0, 1, 0, 1, 1])

    def make_client(self, model=None, batch_size=2):
        if model is None:
            model = RecordingLinear()
        return BaselineExperiment(
            model,
            LocalDataset(self.images, self.labels, batch_size),
            learning_rate=0.1,
            momentum=0.0,
            device="cpu",
            seed=7,
        )

    def test_fit_honors_epochs_with_one_forward_per_batch(self):
        model = RecordingLinear()
        client = self.make_client(model)
        initial_parameters = get_params(model)

        parameters, num_examples, metrics = client.fit(
            initial_parameters, {"epochs": 3, "proximal_mu": 0.0}
        )

        # Five examples in batches of two: three batches per epoch.
        self.assertEqual(len(model.outputs), 9)
        self.assertEqual(num_examples, 5)
        self.assertFalse(np.allclose(parameters[0], initial_parameters[0]))
        expected_loss = F.cross_entropy(
            torch.cat(model.outputs), self.labels.repeat(3)
        ).item()
        self.assertAlmostEqual(metrics["loss"], expected_loss, places=6)

    def test_validation_loss_is_mean_per_example_with_partial_final_batch(self):
        # The last example has a deliberately large error to expose averaging
        # batch means without weighting the final, smaller batch.
        images = torch.tensor(
            [[4.0, 0.0], [0.0, 4.0], [1.0, 0.0], [0.0, 1.0], [9.0, -9.0]]
        )
        labels = torch.tensor([0, 1, 0, 1, 1])
        expected_loss = F.cross_entropy(images, labels).item()

        for batch_size in (2, 3, 5):
            with self.subTest(batch_size=batch_size):
                model = nn.Linear(2, 2, bias=False)
                with torch.no_grad():
                    model.weight.copy_(torch.eye(2))
                dataset = LocalDataset(images, labels, batch_size)
                client = BaselineExperiment(model, dataset, device="cpu")

                loss, num_examples, metrics = client.evaluate(get_params(model), {})

                self.assertAlmostEqual(loss, expected_loss, places=6)
                self.assertEqual(num_examples, 5)
                self.assertAlmostEqual(metrics["accuracy"], 0.8)
                self.assertTrue(all(p.grad is None for p in model.parameters()))

    def test_fedprox_uses_squared_distance_to_fixed_global_weights(self):
        images = torch.ones(2, 1)
        labels = torch.ones(2, dtype=torch.long)
        mu = 2.0
        learning_rate = 0.1
        results = {}
        for proximal_mu in (0.0, mu):
            model = nn.Linear(1, 2, bias=False)
            nn.init.zeros_(model.weight)
            client = BaselineExperiment(
                model,
                LocalDataset(images, labels, batch_size=1),
                learning_rate=learning_rate,
                momentum=0.0,
                device="cpu",
            )
            parameters, _, _ = client.fit(
                get_params(model), {"epochs": 1, "proximal_mu": proximal_mu}
            )
            results[proximal_mu] = parameters[0].reshape(-1)

        # First step has zero proximal gradient. On step two the derivative
        # of (mu/2) * ||w - w_global||^2 is mu * (w - w_global).
        first_step = torch.tensor([-0.05, 0.05])
        data_gradient = torch.softmax(first_step, dim=0) - torch.tensor([0.0, 1.0])
        expected = first_step - learning_rate * (data_gradient + mu * first_step)
        np.testing.assert_allclose(results[mu], expected.numpy(), atol=1e-7)
        self.assertLess(np.linalg.norm(results[mu]), np.linalg.norm(results[0.0]))

    def test_zero_mu_matches_fedavg(self):
        model = RecordingLinear()
        initial_parameters = get_params(model)
        results = []
        for config in ({"epochs": 2}, {"epochs": 2, "proximal_mu": 0.0}):
            client = self.make_client()
            parameters, _, _ = client.fit(initial_parameters, config)
            results.append(parameters)
        for fedavg, zero_mu in zip(*results):
            np.testing.assert_allclose(fedavg, zero_mu, atol=1e-7)

    def test_each_fit_uses_new_global_weights_and_resets_optimizer(self):
        reused_client = self.make_client()
        reused_client.fit(
            reused_client.get_parameters({}),
            {"epochs": 2, "proximal_mu": 2.0, "momentum": 0.9, "server_round": 1},
        )
        incoming_model = RecordingLinear()
        with torch.no_grad():
            incoming_model.weight.fill_(0.75)
        incoming_parameters = get_params(incoming_model)
        before_fit = [array.copy() for array in incoming_parameters]
        config = {"epochs": 2, "proximal_mu": 2.0, "momentum": 0.9, "server_round": 2}

        reused_parameters, _, _ = reused_client.fit(incoming_parameters, config)
        fresh_parameters, _, _ = self.make_client().fit(incoming_parameters, config)

        for reused, fresh in zip(reused_parameters, fresh_parameters):
            np.testing.assert_allclose(reused, fresh, atol=1e-7)
        for received, original in zip(incoming_parameters, before_fit):
            np.testing.assert_array_equal(received, original)

    def test_model_parameters_roundtrip(self):
        source = nn.Linear(2, 2)
        target = nn.Linear(2, 2)
        parameters = get_params(source)

        set_params(target, parameters)

        for received, expected in zip(get_params(target), parameters):
            np.testing.assert_array_equal(received, expected)


if __name__ == "__main__":
    unittest.main()
