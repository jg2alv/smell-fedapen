"""Flower client for the three stages of the SMELL experiment."""

from collections import OrderedDict
from io import BytesIO
import random

import numpy as np
import torch
from flwr.client import NumPyClient
from flwr.common import ConfigRecord, RecordDict
from sklearn.neighbors import KNeighborsClassifier
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassF1Score,
    MulticlassPrecision,
    MulticlassRecall,
)


def get_params(model):
    return [value.detach().cpu().numpy().copy() for value in model.state_dict().values()]


def set_params(model, parameters):
    state = OrderedDict(
        (name, torch.as_tensor(array, dtype=current.dtype))
        for (name, current), array in zip(model.state_dict().items(), parameters, strict=True)
    )
    model.load_state_dict(state, strict=True)


class SmellExperiment(NumPyClient):
    def __init__(self, model, dataset, state=None, device="cpu", seed=42, pairs_per_batch=200):
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.dataset = dataset
        self.state = RecordDict() if state is None else state
        self.seed = seed
        self.pairs_per_batch = pairs_per_batch

    def get_parameters(self, config):
        return get_params(self.model)

    def create_optimizer(self, phase):
        if phase == "pretrain":
            parameters = list(self.model.encoder.parameters()) + list(self.model.center_loss.parameters())
            optimizer = torch.optim.Adam(parameters, lr=0.001)
            scheduler = None
        else:
            if phase == "prototypes":
                optimizer = torch.optim.Adam(self.model.pair_loss.parameters(), lr=0.01)
            elif phase == "joint":
                parameters = list(self.model.encoder.parameters()) + list(self.model.pair_loss.parameters())
                optimizer = torch.optim.SGD(parameters, lr=0.01)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)
        saved = self.state.get("optimizer")
        if saved is not None and saved["phase"] == phase:
            payload = torch.load(BytesIO(saved["state"]), map_location="cpu", weights_only=True)
            optimizer.load_state_dict(payload["optimizer"])
            if scheduler is not None:
                scheduler.load_state_dict(payload["scheduler"])
        return optimizer, scheduler

    def save_optimizer(self, phase, optimizer, scheduler):
        # Flower recreates clients between messages; retain optimizer moments
        # and scheduler progress in the client's in-memory Context.state.
        buffer = BytesIO()
        torch.save({
            "optimizer": optimizer.state_dict(),
            "scheduler": None if scheduler is None else scheduler.state_dict(),
        }, buffer)
        self.state["optimizer"] = ConfigRecord({"phase": phase, "state": buffer.getvalue()})

    def fit(self, parameters, config):
        phase = config["phase"]
        seed = self.seed + config["server_round"]
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        set_params(self.model, parameters)
        optimizer, scheduler = self.create_optimizer(phase)
        self.model.train()
        total_loss, examples, positive_pairs = 0.0, 0, 0
        if phase == "pretrain":
            loader = self.dataset.dataloader(seed=seed)
        else:
            loader = self.dataset.pair_loader(pairs_per_batch=self.pairs_per_batch, seed=seed)
        for epoch in range(config["epochs"]):
            if phase != "pretrain":
                loader.dataset.epoch = epoch
            for batch in loader:
                optimizer.zero_grad(set_to_none=True)
                if phase == "pretrain":
                    images = batch["image"].to(self.device)
                    labels = batch["label"].to(self.device)
                    loss = self.model.center_loss(self.model(images), labels)
                else:
                    first = batch["first"].to(self.device)
                    second = batch["second"].to(self.device)
                    labels = batch["similar"].to(self.device)
                    if phase == "prototypes":
                        with torch.no_grad():
                            first_embedding = self.model(first)
                            second_embedding = self.model(second)
                    else:
                        first_embedding = self.model(first)
                        second_embedding = self.model(second)
                    loss = self.model.pair_loss(first_embedding, second_embedding, labels)
                    positive_pairs += labels.sum().item()
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(labels)
                examples += len(labels)
            if scheduler is not None:
                scheduler.step()
        self.save_optimizer(phase, optimizer, scheduler)
        metrics = {"loss": total_loss / examples, "learning_rate": optimizer.param_groups[0]["lr"]}
        if phase != "pretrain":
            metrics["positive_pair_fraction"] = positive_pairs / examples
        # FedAvg weights updates by original images, not resampled pair counts.
        return get_params(self.model), len(self.dataset.train_partition), metrics

    def embeddings(self, validation=False):
        features, labels = [], []
        with torch.no_grad():
            for batch in self.dataset.dataloader(validation=validation, shuffle=False):
                features.append(self.model(batch["image"].to(self.device)).cpu())
                labels.append(batch["label"])
        return torch.cat(features).numpy(), torch.cat(labels)

    def evaluate(self, parameters, config):
        set_params(self.model, parameters)
        self.model.eval()
        train_embeddings, train_labels = self.embeddings()
        test_embeddings, test_labels = self.embeddings(validation=True)
        classifier = KNeighborsClassifier(n_neighbors=1)
        classifier.fit(train_embeddings, train_labels.numpy())
        predicted = torch.from_numpy(classifier.predict(test_embeddings))
        phase = config["phase"]
        total_loss, examples, positive_pairs = 0.0, 0, 0
        with torch.no_grad():
            if phase == "pretrain":
                for batch in self.dataset.dataloader(validation=True):
                    labels = batch["label"].to(self.device)
                    loss = self.model.center_loss(self.model(batch["image"].to(self.device)), labels)
                    total_loss += loss.item() * len(labels)
                    examples += len(labels)
            else:
                loader = self.dataset.pair_loader(
                    validation=True, pairs_per_batch=self.pairs_per_batch, seed=self.seed + 1_000_000,
                )
                for batch in loader:
                    labels = batch["similar"].to(self.device)
                    loss = self.model.pair_loss(
                        self.model(batch["first"].to(self.device)),
                        self.model(batch["second"].to(self.device)), labels,
                    )
                    total_loss += loss.item() * len(labels)
                    examples += len(labels)
                    positive_pairs += labels.sum().item()
        metric_classes = {
            "accuracy": MulticlassAccuracy,
            "precision": MulticlassPrecision,
            "recall": MulticlassRecall,
            "f1": MulticlassF1Score,
        }
        metrics = {
            name: float(metric(num_classes=10, average="macro")(predicted, test_labels))
            for name, metric in metric_classes.items()
        }
        metrics["center_loss" if phase == "pretrain" else "smell_loss"] = total_loss / examples
        if phase != "pretrain":
            metrics["positive_pair_fraction"] = positive_pairs / examples
        return total_loss / examples, len(self.dataset.test_partition), metrics
