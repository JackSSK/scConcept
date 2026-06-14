from inspect import signature
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pandas as pd
from omegaconf import OmegaConf

from concept.api import scConcept


def _minimal_training_config(tmp_path):
    return OmegaConf.create(
        {
            "PATH": {
                "PANELS_PATH": str(tmp_path / "panels"),
                "GENE_MAPPINGS_PATH": str(tmp_path / "gene_mappings"),
            },
            "datamodule": {
                "species": ["hsapiens"],
                "precomp_embs_key": None,
                "normalization": "raw",
                "gene_sampling_strategy": "top-nonzero",
                "dataset": {
                    "train": {"split": None},
                    "val": None,
                },
                "dataloader": {
                    "train": {"batch_size": 4},
                    "val": None,
                },
            },
            "model": {
                "training": {
                    "max_steps": 1,
                    "limit_train_batches": 1.0,
                    "accumulate_grad_batches": 1,
                    "log_every_n_steps": 1,
                }
            },
        }
    )


def test_train_uses_csv_logger_save_dir(tmp_path):
    concept = scConcept(cfg=_minimal_training_config(tmp_path))
    save_dir = tmp_path / "training_logs"
    panels_dir = tmp_path / "custom_panels"
    csv_logger = MagicMock()
    csv_logger.log_dir = str(save_dir / "lightning_logs" / "version_0")
    trainer = MagicMock()
    tokenizer = SimpleNamespace(PAD_TOKEN=1, CLS_TOKEN=0, vocab_sizes={"hsapiens": 10})

    with (
        patch("concept.api.build_species_gene_mappings", return_value={}),
        patch("concept.api.MultiSpeciesTokenizer", return_value=tokenizer),
        patch("concept.api.AnnDataModule", return_value=MagicMock()) as datamodule_cls,
        patch("concept.api.ContrastiveModel", return_value=MagicMock()),
        patch("concept.api.CSVLogger", return_value=csv_logger) as csv_logger_cls,
        patch("concept.api.L.Trainer", return_value=trainer) as trainer_cls,
    ):
        concept.train(
            adata_list=str(tmp_path / "data.h5ad"),
            species="hsapiens",
            save_dir=save_dir,
            panels_dir=panels_dir,
        )

    csv_logger_cls.assert_called_once_with(save_dir=save_dir)
    assert datamodule_cls.call_args.kwargs["panels_path"] == panels_dir
    assert trainer_cls.call_args.kwargs["logger"] is csv_logger
    assert concept.training_log_dir == save_dir / "lightning_logs" / "version_0"
    assert concept.training_metrics_path == save_dir / "lightning_logs" / "version_0" / "metrics.csv"
    trainer.fit.assert_called_once()


def test_train_save_dir_defaults_to_training_logs():
    default_save_dir = signature(scConcept.train).parameters["save_dir"].default

    assert default_save_dir == "./training_logs/"


def test_train_panels_dir_defaults_to_none():
    default_panels_dir = signature(scConcept.train).parameters["panels_dir"].default

    assert default_panels_dir is None


def test_plot_training_curves_uses_current_metrics_file(tmp_path):
    import matplotlib

    matplotlib.use("Agg")

    metrics_path = tmp_path / "lightning_logs" / "version_0" / "metrics.csv"
    metrics_path.parent.mkdir(parents=True)
    pd.DataFrame(
        {
            "step": [0, 0, 1, 1],
            "train/loss": [1.0, None, 0.75, None],
            "train/recall@1": [None, 0.2, None, 0.45],
        }
    ).to_csv(metrics_path, index=False)

    concept = scConcept()
    concept.training_metrics_path = metrics_path
    save_path = tmp_path / "curves.png"

    fig, axes = concept.plot_training_curves(save_path=save_path, show=False)

    assert len(axes) == 2
    assert axes[0].get_ylabel() == "train/loss"
    assert axes[1].get_ylabel() == "train/recall@1"
    assert save_path.exists()
    fig.clf()


def test_plot_training_curves_averages_logged_metric_entries(tmp_path):
    import matplotlib

    matplotlib.use("Agg")

    metrics_path = tmp_path / "metrics.csv"
    pd.DataFrame(
        {
            "step": [0, 0, 1, 1, 2, 2, 3, 3],
            "train/loss": [10.0, None, 8.0, None, 6.0, None, 4.0, None],
            "train/recall@1": [None, 0.1, None, 0.3, None, 0.5, None, 0.7],
        }
    ).to_csv(metrics_path, index=False)

    concept = scConcept()

    fig, axes = concept.plot_training_curves(metrics_path=metrics_path, show=False, n_avg=2)

    loss_line = axes[0].lines[0]
    recall_line = axes[1].lines[0]
    assert loss_line.get_xdata().tolist() == [0.5, 2.5]
    assert loss_line.get_ydata().tolist() == [9.0, 5.0]
    assert recall_line.get_xdata().tolist() == [0.5, 2.5]
    assert recall_line.get_ydata().tolist() == [0.2, 0.6]
    fig.clf()
