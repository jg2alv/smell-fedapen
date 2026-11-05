"""Native PyTorch training and evaluation exposed as a Flower NumPyClient."""

from collections import OrderedDict
import math
import random

import numpy as np
import torch
from flwr.client import NumPyClient
from sklearn.metrics import f1_score


def set_params(model, parameters):
    """Load Flower arrays, validating the entire state before updating it."""
    state = model.state_dict()
    if len(parameters) != len(state):
        raise ValueError("The number of parameter arrays does not match the model.")
    updated = OrderedDict()
    for (name, current), array in zip(state.items(), parameters):
        value = torch.as_tensor(array, dtype=current.dtype)
        if value.shape != current.shape:
            raise ValueError(f"Invalid shape for parameter {name}: {value.shape}.")
        updated[name] = value
    model.load_state_dict(updated, strict=True)


def get_params(model):
    """Return independent CPU arrays suitable for transport by Flower."""
    return [value.detach().cpu().numpy().copy() for value in model.state_dict().values()]


def _positive_epochs(epochs):
    if isinstance(epochs, bool) or not isinstance(epochs, (int, np.integer)) or epochs < 1:
        raise ValueError("epochs must be a positive integer.")
    return int(epochs)


def _validate_optimizer(learning_rate, momentum):
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be finite and greater than zero.")
    if not math.isfinite(momentum) or not 0 <= momentum < 1:
        raise ValueError("momentum must be finite and in [0, 1).")


class PoCExperimentFMNIST(NumPyClient):
    """Fashion-MNIST client supporting FedAvg and the FedProx local objective."""

    def __init__(
        self,
        model,
        dataset,
        *,
        learning_rate=0.01,
        momentum=0.9,
        epochs=1,
        device="cpu",
        seed=42,
    ):
        super().__init__()
        _validate_optimizer(learning_rate, momentum)
        self.epochs = _positive_epochs(epochs)
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.dataset = dataset
        self.learning_rate = float(learning_rate)
        self.momentum = float(momentum)
        self.seed = int(seed)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(
            self.model.parameters(), lr=self.learning_rate, momentum=self.momentum
        )

    def get_parameters(self, config):
        return get_params(self.model)

    def fit(self, parameters, config):
        """Train from this round's global weights; Flower FedProx supplies mu."""
        epochs = _positive_epochs(config.get("epochs", self.epochs))
        learning_rate = float(config.get("learning_rate", self.learning_rate))
        momentum = float(config.get("momentum", self.momentum))
        proximal_mu = float(config.get("proximal_mu", 0.0))
        _validate_optimizer(learning_rate, momentum)
        if not math.isfinite(proximal_mu) or proximal_mu < 0:
            raise ValueError("proximal_mu must be finite and nonnegative.")

        round_seed = int(config.get("seed", self.seed)) + int(config.get("server_round", 0))
        random.seed(round_seed)
        np.random.seed(round_seed % (2**32))
        torch.manual_seed(round_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(round_seed)

        set_params(self.model, parameters)
        # These detached weights stay fixed throughout every local epoch.
        global_parameters = tuple(weight.detach().clone() for weight in self.model.parameters())
        # Clients can be recreated between rounds; momentum always starts fresh.
        self.optimizer = torch.optim.SGD(
            self.model.parameters(), lr=learning_rate, momentum=momentum
        )
        _, metrics = self.training_loop(
            self.dataset.dataloader(),
            epochs=epochs,
            proximal_mu=proximal_mu,
            global_parameters=global_parameters,
        )
        return get_params(self.model), len(self.dataset.train_partition), metrics

    def training_loop(self, data_loader, *, epochs=None, proximal_mu=0.0, global_parameters=None):
        """Minimize CE + mu/2 * ||w-w_global||² over all requested epochs."""
        epochs = _positive_epochs(self.epochs if epochs is None else epochs)
        if not math.isfinite(proximal_mu) or proximal_mu < 0:
            raise ValueError("proximal_mu must be finite and nonnegative.")
        if proximal_mu > 0:
            if global_parameters is None:
                raise ValueError("FedProx requires a fixed snapshot of the global parameters.")
            global_parameters = tuple(weight.detach().to(self.device) for weight in global_parameters)
            if len(global_parameters) != len(tuple(self.model.parameters())):
                raise ValueError("The FedProx snapshot does not match the model.")

        self.model.train()
        total_loss, correct, num_examples = 0.0, 0, 0
        for _ in range(epochs):
            for batch in data_loader:
                images = batch["image"].to(self.device)
                labels = batch["label"].to(self.device)
                self.optimizer.zero_grad(set_to_none=True)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                if proximal_mu > 0:
                    proximal_term = sum(
                        (local - global_weight).square().sum()
                        for local, global_weight in zip(self.model.parameters(), global_parameters)
                    )
                    loss = loss + (proximal_mu / 2.0) * proximal_term
                loss.backward()
                self.optimizer.step()

                batch_size = labels.size(0)
                total_loss += loss.item() * batch_size
                correct += (outputs.argmax(dim=1) == labels).sum().item()
                num_examples += batch_size

        if num_examples == 0:
            raise ValueError("The training data loader is empty.")
        loss = total_loss / num_examples
        return float(loss), {"loss": float(loss), "accuracy": correct / num_examples}

    def evaluate(self, parameters, config):
        set_params(self.model, parameters)
        loss, metrics = self.validation_loop(self.dataset.dataloader(validation=True))
        return loss, len(self.dataset.test_partition), metrics

    def validation_loop(self, data_loader):
        """Return sample-weighted mean CE, accuracy and macro F1, without mu."""
        self.model.eval()
        total_loss, correct, num_examples = 0.0, 0, 0
        predictions, labels_all = [], []
        with torch.no_grad():
            for batch in data_loader:
                images = batch["image"].to(self.device)
                labels = batch["label"].to(self.device)
                outputs = self.model(images)
                batch_size = labels.size(0)
                total_loss += self.criterion(outputs, labels).item() * batch_size
                predicted = outputs.argmax(dim=1)
                correct += (predicted == labels).sum().item()
                num_examples += batch_size
                predictions.extend(predicted.cpu().tolist())
                labels_all.extend(labels.cpu().tolist())

        if num_examples == 0:
            raise ValueError("The validation data loader is empty.")
        loss = total_loss / num_examples
        return float(loss), {
            "loss": float(loss),
            "accuracy": correct / num_examples,
            "f1": float(f1_score(labels_all, predictions, average="macro", zero_division=0)),
        }
