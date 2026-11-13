"""Offline checks for charts generated from training and evaluation histories."""

from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from matplotlib.axes import Axes
from matplotlib.figure import Figure

from plots import plot_history, plt


class PlotHistoryTests(unittest.TestCase):
    def setUp(self):
        self.history = SimpleNamespace(
            metrics_distributed_fit={
                "accuracy_private": [(1, 0.7), (3, 0.8)],
                "accuracy_shared": [(1, 0.6), (3, 0.75)],
                "loss_private": [(1, 0.8), (3, 0.4)],
                "loss_shared": [(1, 0.9), (3, 0.5)],
            },
            metrics_distributed={
                "accuracy": [(1, 0.75), (3, 0.85)],
                "loss": [(1, 0.7), (3, 0.3)],
                "lambda": [(1, 0.5), (3, 0.6)],
                "f1": [(1, 0.65), (3, 0.8)],
                "bmta": [(1, 0.75), (3, 0.85)],
            },
            losses_distributed=[(1, 0.7), (3, 0.3)],
        )

    def test_generates_one_png_per_metric_and_stage(self):
        with TemporaryDirectory() as directory:
            paths = plot_history(self.history, Path(directory) / "charts")
            expected = {
                "treino_accuracy_private.png", "treino_accuracy_shared.png",
                "treino_loss_private.png", "treino_loss_shared.png",
                "avaliacao_accuracy.png", "avaliacao_loss.png",
                "avaliacao_lambda.png", "avaliacao_f1.png", "avaliacao_bmta.png",
            }
            self.assertEqual({path.name for path in paths}, expected)
            self.assertEqual(len(paths), len(expected))
            for path in paths:
                self.assertEqual(path.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
            self.assertEqual(plt.get_fignums(), [])

    def test_preserves_recorded_rounds_and_values_without_smoothing(self):
        self.history.metrics_distributed_fit = {"loss": [(1, 0.9), (4, 0.25), (9, 0.6)]}
        self.history.metrics_distributed = {}
        self.history.losses_distributed = []
        recorded = []
        original_plot = Axes.plot

        def record_plot(ax, rounds, values, *args, **kwargs):
            recorded.append((tuple(rounds), tuple(values)))
            return original_plot(ax, rounds, values, *args, **kwargs)

        with TemporaryDirectory() as directory, patch.object(Axes, "plot", record_plot):
            plot_history(self.history, directory)
        self.assertEqual(recorded, [((1, 4, 9), (0.9, 0.25, 0.6))])

    def test_uses_distributed_loss_when_not_present_in_metrics(self):
        self.history.metrics_distributed_fit = {}
        self.history.metrics_distributed = {}
        with TemporaryDirectory() as directory:
            paths = plot_history(self.history, directory)
            self.assertEqual([path.name for path in paths], ["avaliacao_loss.png"])

    def test_uses_metric_loss_once_when_also_available_as_distributed_loss(self):
        self.history.metrics_distributed_fit = {}
        self.history.metrics_distributed = {"loss": [(1, 0.1)]}
        recorded = []
        original_plot = Axes.plot

        def record_plot(ax, rounds, values, *args, **kwargs):
            recorded.append(tuple(values))
            return original_plot(ax, rounds, values, *args, **kwargs)

        with TemporaryDirectory() as directory, patch.object(Axes, "plot", record_plot):
            paths = plot_history(self.history, directory)
        self.assertEqual([path.name for path in paths], ["avaliacao_loss.png"])
        self.assertEqual(recorded, [(0.1,)])

    def test_only_exports_available_series_without_evaluation(self):
        self.history.metrics_distributed_fit = {"loss": [(1, 0.9)], "accuracy": []}
        self.history.metrics_distributed = {}
        self.history.losses_distributed = []
        with TemporaryDirectory() as directory:
            paths = plot_history(self.history, directory)
            self.assertEqual([path.name for path in paths], ["treino_loss.png"])

    def test_propagates_save_errors_and_closes_figures(self):
        with TemporaryDirectory() as directory:
            with patch.object(Figure, "savefig", side_effect=OSError("Cannot save chart")):
                with self.assertRaisesRegex(OSError, "Cannot save chart"):
                    plot_history(self.history, directory)
        self.assertEqual(plt.get_fignums(), [])


if __name__ == "__main__":
    unittest.main()
