"""Offline checks for Fashion-MNIST preprocessing and the native MLP."""

import unittest

import numpy as np
import torch
from datasets import Dataset, Features, Image as ImageFeature, Value
from PIL import Image

from FMNISTDataset import FMNISTDataset
from PoCModel_2HiddenLayers import PoCModel_2HiddenLayers


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
        first = FMNISTDataset(self.partition, seed=17, validation_fraction=0.2)
        second = FMNISTDataset(self.partition, seed=17, validation_fraction=0.2)
        train_ids = list(first.train_partition["sample_id"])
        validation_ids = list(first.test_partition["sample_id"])

        self.assertEqual(train_ids, list(second.train_partition["sample_id"]))
        self.assertEqual(validation_ids, list(second.test_partition["sample_id"]))
        self.assertEqual(len(train_ids), 20)
        self.assertEqual(len(validation_ids), 5)
        self.assertFalse(set(train_ids) & set(validation_ids))
        self.assertEqual(set(train_ids) | set(validation_ids), set(range(25)))

    def test_batches_preserve_labels_and_image_dimensions(self):
        dataset = FMNISTDataset(self.partition, batch_size=4, shuffle=False, augment=False)
        batch = next(iter(dataset.dataloader()))

        self.assertEqual(tuple(batch["image"].shape), (4, 1, 28, 28))
        self.assertEqual(batch["image"].dtype, torch.float32)
        self.assertEqual(batch["label"].dtype, torch.int64)
        self.assertEqual(batch["label"].tolist(), list(dataset.train_partition["label"][:4]))
        self.assertTrue(torch.isfinite(batch["image"]).all())

    def test_batch_size_and_worker_configuration_are_respected(self):
        for workers in (0, 1):
            with self.subTest(num_workers=workers):
                dataset = FMNISTDataset(
                    self.partition, batch_size=7, num_workers=workers, shuffle=False, augment=False
                )
                loader = dataset.dataloader()
                self.assertEqual(loader.batch_size, 7)
                self.assertEqual(loader.num_workers, workers)
                sizes = [len(batch["label"]) for batch in loader]
                self.assertEqual(sizes, [7, 7, 6])

    def test_central_evaluation_uses_the_complete_test_partition(self):
        dataset = FMNISTDataset.from_test_partition(self.partition, batch_size=8, seed=23)
        batches = list(dataset.dataloader(validation=True))
        labels = torch.cat([batch["label"] for batch in batches]).tolist()

        self.assertEqual(len(dataset.test_partition), 25)
        self.assertEqual(labels, list(self.partition["label"]))
        self.assertEqual([len(batch["label"]) for batch in batches], [8, 8, 8, 1])
        with self.assertRaises(ValueError):
            dataset.dataloader()

    def test_disabling_augmentation_is_deterministic(self):
        dataset = FMNISTDataset(self.partition, augment=False, shuffle=False)
        torch.manual_seed(1)
        first = dataset.train()[0]["image"]
        torch.manual_seed(999)
        second = dataset.train()[0]["image"]
        torch.testing.assert_close(first, second, rtol=0, atol=0)

        raw = torch.tensor(np.array(dataset.train_partition[0]["image"]), dtype=torch.float32)
        expected = (raw.unsqueeze(0) / 255.0 - 0.1307) / 0.3081
        torch.testing.assert_close(first, expected)

    def test_seed_reproduces_training_shuffle(self):
        first = FMNISTDataset(self.partition, seed=71, batch_size=6, augment=False)
        second = FMNISTDataset(self.partition, seed=71, batch_size=6, augment=False)
        first_order = torch.cat([batch["sample_id"] for batch in first.dataloader()])
        second_order = torch.cat([batch["sample_id"] for batch in second.dataloader()])
        torch.testing.assert_close(first_order, second_order, rtol=0, atol=0)

    def test_model_returns_ten_logits_per_image(self):
        model = PoCModel_2HiddenLayers()
        with torch.no_grad():
            outputs = model(torch.zeros(5, 1, 28, 28))
        self.assertEqual(tuple(outputs.shape), (5, 10))
        self.assertTrue(torch.isfinite(outputs).all())


if __name__ == "__main__":
    unittest.main()
