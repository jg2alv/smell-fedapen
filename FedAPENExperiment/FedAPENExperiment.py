"""Persistent private models and adaptive ensembles for a Flower client."""

from collections import OrderedDict
import random

import numpy as np
import torch
from flwr.client import NumPyClient
from flwr.common import ArrayRecord, ConfigRecord
from sklearn.metrics import f1_score
from torch.nn import functional as F


def get_params(model):
    return [value.detach().cpu().numpy().copy() for value in model.state_dict().values()]


def set_params(model, parameters):
    updated = OrderedDict(
        (name, torch.as_tensor(array, dtype=current.dtype))
        for (name, current), array in zip(model.state_dict().items(), parameters, strict=True)
    )
    model.load_state_dict(updated, strict=True)


class FedAPENExperiment(NumPyClient):
    def __init__(
        self, shared_model, private_model, dataset, state, *, learning_rate=0.01,
        momentum=0.0, optimizer="sgd", epochs=5, adaptability_epochs=10,
        adaptability_learning_rate=0.001, lambda_init=0.5, device="cpu", seed=42,
    ):
        super().__init__()
        self.device = torch.device(device)
        self.shared_model = shared_model.to(self.device)
        self.private_model = private_model.to(self.device)
        self.dataset = dataset
        self.state = state
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.optimizer_name = optimizer
        self.epochs = epochs
        self.adaptability_epochs = adaptability_epochs
        self.adaptability_learning_rate = adaptability_learning_rate
        self.seed = seed
        if "fedapen.private_model" in state:
            self.private_model.load_state_dict(state["fedapen.private_model"].to_torch_state_dict())
        lambda_value = lambda_init
        if "fedapen.personalization" in state:
            lambda_value = state["fedapen.personalization"]["lambda_i"]
        self.lambda_i = torch.tensor([lambda_value], device=self.device, requires_grad=True)
        self.shared_optimizer = self.create_optimizer(self.shared_model, learning_rate, momentum)
        self.private_optimizer = self.create_optimizer(self.private_model, learning_rate, momentum)
        self.lambda_optimizer = torch.optim.SGD([self.lambda_i], lr=adaptability_learning_rate)
        self.save_state()

    def create_optimizer(self, model, learning_rate, momentum):
        if self.optimizer_name == "sgd":
            return torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
        if self.optimizer_name == "adam":
            if momentum != 0:
                raise ValueError("momentum is only supported by SGD; use --momentum 0 with Adam.")
            return torch.optim.Adam(model.parameters(), lr=learning_rate)
        raise ValueError(f"Unknown optimizer: {self.optimizer_name}")

    def save_state(self):
        private_state = OrderedDict(
            (name, value.detach().cpu().clone())
            for name, value in self.private_model.state_dict().items()
        )
        self.state["fedapen.private_model"] = ArrayRecord(private_state)
        self.state["fedapen.personalization"] = ConfigRecord({"lambda_i": self.lambda_i.item()})

    def seed_round(self, config):
        seed = self.seed + int(config.get("server_round", 0))
        random.seed(seed)
        np.random.seed(seed % (2**32))
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def get_parameters(self, config):
        return get_params(self.shared_model)

    def fit(self, parameters, config):
        self.seed_round(config)
        set_params(self.shared_model, parameters)
        learning_rate = config.get("learning_rate", self.learning_rate)
        momentum = config.get("momentum", self.momentum)
        self.shared_optimizer = self.create_optimizer(self.shared_model, learning_rate, momentum)
        self.private_optimizer = self.create_optimizer(self.private_model, learning_rate, momentum)
        metrics = self.training_loop(
            self.dataset.train_dataloader(), config.get("epochs", self.epochs),
        )
        self.save_state()
        return get_params(self.shared_model), len(self.dataset.train_partition), metrics

    def training_loop(self, data_loader, epochs):
        self.private_model.train()
        self.shared_model.train()
        weight = self.lambda_i.detach()
        private_loss_sum, shared_loss_sum = 0.0, 0.0
        private_correct, shared_correct, num_examples = 0, 0, 0
        for _ in range(epochs):
            for batch in data_loader:
                images = batch["image"].to(self.device)
                labels = batch["label"].to(self.device)
                self.private_optimizer.zero_grad(set_to_none=True)
                self.shared_optimizer.zero_grad(set_to_none=True)
                private_logits = self.private_model(images)
                shared_logits = self.shared_model(images)
                private_ensemble = weight * private_logits + (1 - weight) * shared_logits.detach()
                shared_ensemble = weight * private_logits.detach() + (1 - weight) * shared_logits
                private_loss = (
                    F.cross_entropy(private_logits, labels)
                    + F.kl_div(F.log_softmax(private_logits, dim=1),
                               F.softmax(shared_logits.detach(), dim=1), reduction="batchmean")
                    + F.cross_entropy(private_ensemble, labels)
                )
                shared_loss = (
                    F.cross_entropy(shared_logits, labels)
                    + F.kl_div(F.log_softmax(shared_logits, dim=1),
                               F.softmax(private_logits.detach(), dim=1), reduction="batchmean")
                    + F.cross_entropy(shared_ensemble, labels)
                )
                private_loss.backward()
                shared_loss.backward()
                self.private_optimizer.step()
                self.shared_optimizer.step()
                batch_size = labels.size(0)
                private_loss_sum += private_loss.item() * batch_size
                shared_loss_sum += shared_loss.item() * batch_size
                private_correct += (private_logits.argmax(1) == labels).sum().item()
                shared_correct += (shared_logits.argmax(1) == labels).sum().item()
                num_examples += batch_size
        return {
            "loss_private": private_loss_sum / num_examples,
            "loss_shared": shared_loss_sum / num_examples,
            "accuracy_private": private_correct / num_examples,
            "accuracy_shared": shared_correct / num_examples,
        }

    def learn_lambda(self, config):
        self.private_model.eval()
        self.shared_model.eval()
        self.lambda_optimizer = torch.optim.SGD(
            [self.lambda_i], lr=config.get("adaptability_learning_rate", self.adaptability_learning_rate),
        )
        loader = self.dataset.adaptability_dataloader()
        for _ in range(config.get("adaptability_epochs", self.adaptability_epochs)):
            for batch in loader:
                images = batch["image"].to(self.device)
                labels = batch["label"].to(self.device)
                with torch.no_grad():
                    private_logits = self.private_model(images)
                    shared_logits = self.shared_model(images)
                self.lambda_optimizer.zero_grad(set_to_none=True)
                logits = self.lambda_i * private_logits + (1 - self.lambda_i) * shared_logits
                F.cross_entropy(logits, labels).backward()
                self.lambda_optimizer.step()
                with torch.no_grad():
                    self.lambda_i.clamp_(0.0, 1.0)

    def evaluate(self, parameters, config):
        self.seed_round(config)
        set_params(self.shared_model, parameters)
        self.learn_lambda(config)
        self.save_state()
        loss_sum, num_examples = 0.0, 0
        predictions, all_labels = [], []
        with torch.no_grad():
            for batch in self.dataset.test_dataloader():
                images = batch["image"].to(self.device)
                labels = batch["label"].to(self.device)
                logits = (
                    self.lambda_i * self.private_model(images)
                    + (1 - self.lambda_i) * self.shared_model(images)
                )
                loss_sum += F.cross_entropy(logits, labels).item() * labels.size(0)
                num_examples += labels.size(0)
                predictions.extend(logits.argmax(1).cpu().tolist())
                all_labels.extend(labels.cpu().tolist())
        loss = loss_sum / num_examples
        accuracy = sum(pred == label for pred, label in zip(predictions, all_labels)) / num_examples
        return loss, num_examples, {
            "loss": loss, "accuracy": accuracy,
            "f1": float(f1_score(all_labels, predictions, average="macro", zero_division=0)),
            "lambda": self.lambda_i.item(),
        }
