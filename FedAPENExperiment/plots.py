"""Save one line chart per metric from a Flower training history."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, PercentFormatter


def plot_history(history, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    evaluation = dict(history.metrics_distributed)
    evaluation.setdefault("loss", history.losses_distributed)
    stages = {
        "treino": history.metrics_distributed_fit,
        "avaliacao": evaluation,
    }
    paths = []
    for stage, metrics in stages.items():
        for metric, series in sorted(metrics.items()):
            if not series:
                continue
            rounds, values = zip(*series)
            fig, ax = plt.subplots(figsize=(8, 4.5))
            try:
                ax.plot(rounds, values, linewidth=1.8, marker="o", markersize=2.5)
                label = "Treinamento" if stage == "treino" else "Avaliação"
                ax.set_title(f"{metric} — {label}")
                ax.set_xlabel("Rodada de treinamento")
                ax.set_ylabel(metric)
                ax.xaxis.set_major_locator(MaxNLocator(integer=True))
                if metric.startswith("accuracy") or metric == "bmta":
                    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
                    ax.set_ylabel("Acurácia (%)")
                ax.grid(alpha=0.25)
                fig.tight_layout()
                path = directory / f"{stage}_{metric}.png"
                fig.savefig(path, dpi=200)
                paths.append(path)
            finally:
                plt.close(fig)
    return paths
