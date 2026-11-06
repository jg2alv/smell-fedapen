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
        self.assertEqual(args.num_clients, 100)
        self.assertEqual(args.clients_per_round, 3)
        self.assertEqual(args.local_epochs, 1)
        self.assertEqual(args.mu, 0.01)
        self.assertTrue(args.augment)

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

    def test_valid_interval_boundaries_are_accepted(self):
        for fraction in ("0", "1"):
            with self.subTest(fraction_evaluate=fraction):
                args = parse_args([
                    "--mu", "0", "--momentum", "0", "--seed", str(2**32 - 1),
                    "--fraction-evaluate", fraction, "--num-workers", "0",
                ])
                self.assertEqual(args.fraction_evaluate, float(fraction))
                self.assertEqual(args.seed, 2**32 - 1)

    def test_invalid_experiments_fail_with_cli_error(self):
        invalid_arguments = [
            ["--strategy", "custom"],
            ["--dataset", "unknown"],
            ["--num-clients", "0"],
            ["--num-clients", "2", "--clients-per-round", "3"],
            ["--clients-per-round", "0"],
            ["--num-rounds", "0"],
            ["--local-epochs", "0"],
            ["--local-epochs", "-1"],
            ["--local-epochs", "1.5"],
            ["--batch-size", "0"],
            ["--mu", "-0.1"],
            ["--learning-rate", "0"],
            ["--learning-rate", "-1"],
            ["--momentum", "-0.1"],
            ["--momentum", "1"],
            ["--alpha", "0"],
            ["--validation-fraction", "0"],
            ["--validation-fraction", "1"],
            ["--fraction-evaluate", "-0.1"],
            ["--fraction-evaluate", "1.1"],
            ["--seed", "-1"],
            ["--seed", str(2**32)],
            ["--num-workers", "-1"],
        ]
        for argv in invalid_arguments:
            with self.subTest(arguments=argv):
                errors = io.StringIO()
                with redirect_stderr(errors), self.assertRaises(SystemExit) as result:
                    parse_args(argv)
                self.assertEqual(result.exception.code, 2)
                self.assertIn("error:", errors.getvalue())

    def test_nonfinite_float_settings_are_rejected(self):
        float_flags = (
            "--mu", "--learning-rate", "--momentum", "--alpha",
            "--validation-fraction", "--fraction-evaluate",
        )
        for flag in float_flags:
            for value in ("nan", "inf", "-inf"):
                with self.subTest(flag=flag, value=value):
                    with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as result:
                        parse_args([f"{flag}={value}"])
                    self.assertEqual(result.exception.code, 2)


class MetricAggregationTests(unittest.TestCase):
    def test_metrics_are_weighted_by_number_of_examples(self):
        result = weighted_average([
            (2, {"loss": 2.0, "accuracy": 0.5}),
            (6, {"loss": 6.0, "accuracy": 1.0}),
        ])

        self.assertEqual(result, {"loss": 5.0, "accuracy": 0.875})

    def test_empty_or_zero_example_results_have_no_metrics(self):
        self.assertEqual(weighted_average([]), {})
        self.assertEqual(weighted_average([(0, {"loss": 1.0})]), {})


if __name__ == "__main__":
    unittest.main()
