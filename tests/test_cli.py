"""Exercise experiment configuration and metric aggregation."""

from contextlib import redirect_stderr
import io
import unittest

from run import parse_args, weighted_average


class ConfigurationTests(unittest.TestCase):
    def test_default_experiment_uses_fedprox(self):
        args = parse_args([])

        self.assertEqual(args.strategy, "fedprox")
        self.assertEqual(args.dataset, "fashion-mnist")
        self.assertEqual(args.num_clients, 10)
        self.assertEqual(args.clients_per_round, 10)
        self.assertEqual(args.local_epochs, 5)
        self.assertEqual(args.batch_size, 64)
        self.assertEqual(args.mu, 0.001)
        self.assertEqual(args.momentum, 0.0)
        self.assertEqual(args.lr_decay, 0.99)
        self.assertEqual(args.validation_fraction, 1 / 6)
        self.assertEqual(args.fraction_evaluate, 1.0)
        self.assertTrue(args.augment)

    def test_defaults_do_not_change_with_strategy(self):
        defaults = vars(parse_args([])).copy()
        defaults.pop("strategy")
        for strategy in ("fedavg", "fedprox", "fedapen"):
            arguments = vars(parse_args(["--strategy", strategy])).copy()
            arguments.pop("strategy")
            self.assertEqual(arguments, defaults)

    def test_fedapen_respects_explicit_hyperparameters(self):
        args = parse_args([
            "--strategy", "fedapen", "--num-clients", "8", "--clients-per-round", "3",
            "--fraction-evaluate", "0.5", "--momentum", "0.2", "--optimizer", "sgd",
            "--init", "xavier", "--lr-decay", "0.8", "--adaptability-fraction", "0.1",
            "--adaptability-epochs", "4", "--adaptability-learning-rate", "0.02",
            "--lambda-init", "0.3",
        ])
        self.assertEqual(args.clients_per_round, 3)
        self.assertEqual(args.fraction_evaluate, 0.5)
        self.assertEqual(args.momentum, 0.2)
        self.assertEqual(args.optimizer, "sgd")
        self.assertEqual(args.init, "xavier")
        self.assertEqual(args.lr_decay, 0.8)
        self.assertEqual(args.adaptability_fraction, 0.1)
        self.assertEqual(args.adaptability_epochs, 4)
        self.assertEqual(args.adaptability_learning_rate, 0.02)
        self.assertEqual(args.lambda_init, 0.3)

    def test_explicit_fedavg_experiment_preserves_requested_settings(self):
        args = parse_args([
            "--strategy", "fedavg", "--num-clients", "8",
            "--clients-per-round", "2", "--num-rounds", "4",
            "--local-epochs", "5", "--batch-size", "16",
            "--learning-rate", "0.03", "--momentum", "0.5", "--seed", "19",
            "--dataset", "cifar10", "--alpha", "0.8",
            "--validation-fraction", "0.25", "--fraction-evaluate", "0.5",
            "--num-workers", "2", "--no-augmentation",
        ])

        expected = {
            "strategy": "fedavg", "num_clients": 8, "clients_per_round": 2,
            "num_rounds": 4, "local_epochs": 5, "batch_size": 16,
            "learning_rate": 0.03, "momentum": 0.5, "seed": 19,
            "dataset": "cifar10", "alpha": 0.8, "validation_fraction": 0.25,
            "fraction_evaluate": 0.5, "num_workers": 2, "augment": False,
        }
        for name, value in expected.items():
            with self.subTest(setting=name):
                self.assertEqual(getattr(args, name), value)

    def test_invalid_experiments_fail_with_cli_error(self):
        invalid_arguments = [
            ["--strategy", "custom"],
            ["--dataset", "unknown"],
            ["--local-epochs", "1.5"],
            ["--mu", "abc"],
        ]
        for argv in invalid_arguments:
            with self.subTest(arguments=argv):
                errors = io.StringIO()
                with redirect_stderr(errors), self.assertRaises(SystemExit) as result:
                    parse_args(argv)
                self.assertEqual(result.exception.code, 2)
                self.assertIn("error:", errors.getvalue())


class MetricAggregationTests(unittest.TestCase):
    def test_metrics_are_weighted_by_number_of_examples(self):
        result = weighted_average([
            (2, {"loss": 2.0, "accuracy": 0.5}),
            (6, {"loss": 6.0, "accuracy": 1.0}),
        ])

        self.assertEqual(result, {"loss": 5.0, "accuracy": 0.875})


if __name__ == "__main__":
    unittest.main()
