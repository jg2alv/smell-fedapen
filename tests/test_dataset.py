"""Offline checks for baseline preprocessing and dataset-specific CNNs."""

import unittest

import numpy as np
import torch
from datasets import Dataset, Features, Image as ImageFeature, Value
from PIL import Image

from BaselineDataset import BaselineDataset
from BaselineExperiment import BaselineExperiment, get_params
from Model import create_model


class DatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pixels = np.arange(28 * 28, dtype=np.uint16).reshape(28, 28)
        cls.partition = Dataset.from_dict(
            {
                "image": [Image.fromarray(((pixels + index) % 256).astype(np.uint8)) for index in range(25)],
                "label": [index % 10 for index in range(25)],
                "sample_id": list(range(25)),
            },
            features=Features(
                {"image": ImageFeature(), "label": Value("int64"), "sample_id": Value("int64")}
            ),
        )

    def test_split_is_reproducible_disjoint_and_complete(self):
        first = BaselineDataset(self.partition, seed=17, validation_fraction=0.2)
        second = BaselineDataset(self.partition, seed=17, validation_fraction=0.2)
        train_ids = list(first.train_partition["sample_id"])
        validation_ids = list(first.test_partition["sample_id"])

        self.assertEqual(train_ids, list(second.train_partition["sample_id"]))
        self.assertEqual(validation_ids, list(second.test_partition["sample_id"]))
        self.assertEqual(len(train_ids), 20)
        self.assertEqual(len(validation_ids), 5)
        self.assertFalse(set(train_ids) & set(validation_ids))
        self.assertEqual(set(train_ids) | set(validation_ids), set(range(25)))

    def test_batches_preserve_labels_and_image_dimensions(self):
        dataset = BaselineDataset(self.partition, batch_size=4, shuffle=False, augment=False)
        batch = next(iter(dataset.dataloader()))

        self.assertEqual(tuple(batch["image"].shape), (4, 1, 28, 28))
        self.assertEqual(batch["image"].dtype, torch.float32)
        self.assertEqual(batch["label"].dtype, torch.int64)
        self.assertEqual(batch["label"].tolist(), list(dataset.train_partition["label"][:4]))
        self.assertTrue(torch.isfinite(batch["image"]).all())

    def test_batch_size_and_worker_configuration_are_respected(self):
        for workers in (0, 1):
            with self.subTest(num_workers=workers):
                dataset = BaselineDataset(
                    self.partition, batch_size=7, num_workers=workers, shuffle=False, augment=False
                )
                loader = dataset.dataloader()
                self.assertEqual(loader.batch_size, 7)
                self.assertEqual(loader.num_workers, workers)
                sizes = [len(batch["label"]) for batch in loader]
                self.assertEqual(sizes, [7, 7, 6])

    def test_disabling_augmentation_is_deterministic(self):
        dataset = BaselineDataset(self.partition, augment=False, shuffle=False)
        torch.manual_seed(1)
        first = dataset.train()[0]["image"]
        torch.manual_seed(999)
        second = dataset.train()[0]["image"]
        torch.testing.assert_close(first, second, rtol=0, atol=0)

        raw = torch.tensor(np.array(dataset.train_partition[0]["image"]), dtype=torch.float32)
        expected = (raw.unsqueeze(0) / 255.0 - 0.1307) / 0.3081
        torch.testing.assert_close(first, expected)

    def test_seed_reproduces_training_shuffle(self):
        first = BaselineDataset(self.partition, seed=71, batch_size=6, augment=False)
        second = BaselineDataset(self.partition, seed=71, batch_size=6, augment=False)
        first_order = torch.cat([batch["sample_id"] for batch in first.dataloader()])
        second_order = torch.cat([batch["sample_id"] for batch in second.dataloader()])
        torch.testing.assert_close(first_order, second_order, rtol=0, atol=0)

    def test_model_returns_ten_logits_per_image(self):
        model = create_model("fashion-mnist")
        with torch.no_grad():
            outputs = model(torch.zeros(5, 1, 28, 28))
        self.assertEqual(tuple(outputs.shape), (5, 10))
        self.assertTrue(torch.isfinite(outputs).all())


class CIFAR10Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(42)
        cls.partition = Dataset.from_dict(
            {
                "img": [
                    Image.fromarray(rng.integers(0, 256, (32, 32, 3), dtype=np.uint8))
                    for _ in range(25)
                ],
                "label": [index % 10 for index in range(25)],
            },
            features=Features({"img": ImageFeature(), "label": Value("int64")}),
        )

    def test_cifar_preprocessing_preserves_rgb_shape_and_normalizes_channels(self):
        for augment in (False, True):
            with self.subTest(augment=augment):
                dataset = BaselineDataset(
                    self.partition, dataset_name="cifar10", batch_size=4,
                    shuffle=False, augment=augment,
                )
                batch = next(iter(dataset.dataloader()))
                self.assertEqual(tuple(batch["image"].shape), (4, 3, 32, 32))
                self.assertEqual(batch["label"].tolist(), dataset.train_partition["label"][:4])
                self.assertTrue(torch.isfinite(batch["image"]).all())
                if not augment:
                    raw = np.array(dataset.train_partition[0]["image"])
                    expected = torch.from_numpy(raw).permute(2, 0, 1).float() / 255.0
                    torch.testing.assert_close(batch["image"][0], (expected - 0.5) / 0.5)

    def test_cifar_model_trains_and_evaluates_with_fedprox(self):
        dataset = BaselineDataset(
            self.partition, dataset_name="cifar10", batch_size=7, augment=False,
        )
        model = create_model("cifar10")
        client = BaselineExperiment(model, dataset, device="cpu")
        parameters, num_examples, metrics = client.fit(
            get_params(model), {"epochs": 1, "proximal_mu": 0.01},
        )
        self.assertEqual(num_examples, 20)
        self.assertTrue(np.isfinite(metrics["loss"]))
        self.assertEqual(parameters[0].shape, (64, 3, 3, 3))
        loss, num_examples, metrics = client.evaluate(parameters, {})
        self.assertEqual(num_examples, 5)
        self.assertTrue(np.isfinite(loss))
        self.assertTrue(0 <= metrics["accuracy"] <= 1)


if __name__ == "__main__":
    unittest.main()
