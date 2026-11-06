"""Uniform aggregation of shared models and personalized FedAPEN metrics."""

import numpy as np
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.strategy import FedAvg


def mean_metrics(results):
    if not results:
        return {}
    keys = set.union(*(set(metrics) for _, metrics in results))
    return {key: float(np.mean([metrics[key] for _, metrics in results])) for key in sorted(keys)}


class FedAPEN(FedAvg):
    def __init__(self, **kwargs):
        kwargs.setdefault("fit_metrics_aggregation_fn", mean_metrics)
        kwargs.setdefault("evaluate_metrics_aggregation_fn", mean_metrics)
        super().__init__(**kwargs)
        self.best_mean_accuracy = 0.0

    def __repr__(self):
        return f"FedAPEN(accept_failures={self.accept_failures})"

    def aggregate_fit(self, server_round, results, failures):
        if not results or (failures and not self.accept_failures):
            return None, {}
        client_arrays = [parameters_to_ndarrays(result.parameters) for _, result in results]
        averaged = [
            np.mean(np.stack(layer), axis=0, dtype=np.float64).astype(layer[0].dtype)
            for layer in zip(*client_arrays, strict=True)
        ]
        metrics = self.fit_metrics_aggregation_fn([
            (result.num_examples, result.metrics) for _, result in results
        ])
        return ndarrays_to_parameters(averaged), metrics

    def aggregate_evaluate(self, server_round, results, failures):
        if not results or (failures and not self.accept_failures):
            return None, {}
        loss = float(np.mean([result.loss for _, result in results]))
        metrics = self.evaluate_metrics_aggregation_fn([
            (result.num_examples, result.metrics) for _, result in results
        ])
        if "accuracy" in metrics:
            self.best_mean_accuracy = max(self.best_mean_accuracy, metrics["accuracy"])
            metrics["bmta"] = self.best_mean_accuracy
        return loss, metrics
