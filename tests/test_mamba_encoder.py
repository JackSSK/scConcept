"""Smoke tests for the Mamba (1/2/3) encoder backbone.

The Mamba kernels require CUDA + the optional ``mamba_ssm`` / ``causal_conv1d`` packages, so the
forward/backward tests are skipped unless both are available. The pure-wiring tests (default still
builds a transformer; selecting a mamba variant builds a ``MambaEncoder``) run everywhere.
"""

import importlib.util
from unittest.mock import patch

import pytest
import torch

from concept.model import ContrastiveModel
from concept.modules.mamba_encoder import MAMBA_AVAILABLE

MAMBA_VARIANTS = ["mamba", "mamba2", "mamba3"]

requires_mamba_gpu = pytest.mark.skipif(
    not (MAMBA_AVAILABLE and torch.cuda.is_available()),
    reason="requires CUDA and the optional mamba_ssm/causal_conv1d packages",
)


def _model_config(*, encoder_type="transformer", dim_model=128, decoder_head=False):
    """Tiny model config; dim_model=128 keeps Mamba2/3 head-dim divisibility happy."""
    return {
        "encoder_type": encoder_type,
        "flash_attention": False,
        "mamba": {"d_state": 16, "d_conv": 4, "expand": 2, "bidirectional": True},
        "dim_gene_embs": dim_model,
        "dim_model": dim_model,
        "num_head": 4,
        "dim_hid": 2 * dim_model,
        "nlayers": 2,
        "dropout": 0.0,
        "decoder_head": decoder_head,
        "mask_value": -1,
        "cls_value": -2,
        "input_encoding": "rank_encoding",
        "pe_max_len": 64,
        "mlm_loss_weight": 0.0,
        "cont_loss_weight": 1.0,
        "contrastive_loss": "multiclass",
        "logit_scale_init_value": 1.0,
        "loss_switch_step": 1000,
        "values_only_sanity_check": False,
        "data_loading_speed_sanity_check": False,
        "projection_dim": None,
        "training": {
            "masking_rate": 0.0,
            "lr": 1e-3,
            "weight_decay": 0.0,
            "optimizer_class": "AdamW",
            "scheduler": None,
            "freeze_pretrained_vocabulary": None,
            "use_learnable_embs_freq": None,
            "warmup": 0,
            "max_steps": 1,
            "min_lr": 0.0,
        },
    }


def _build_model(config, device="cpu", vocab_size=64):
    model = ContrastiveModel(
        config=config,
        pad_token_id=0,
        cls_token_id=1,
        vocab_sizes={"hsapiens": vocab_size},
        world_size=1,
        val_loader_names=["val_test"],
    ).to(device)
    model.set_active_species("hsapiens")
    return model


def _make_paired_batch(device, vocab_size=64, batch_size=8, seq_len=20):
    panel = torch.randint(2, vocab_size, (1, 10)).repeat(batch_size, 1)
    return {
        "tokens_1": torch.randint(2, vocab_size, (batch_size, seq_len)).to(device),
        "values_1": torch.randn(batch_size, seq_len).to(device),
        "panel_1": panel.to(device),
        "tokens_2": torch.randint(2, vocab_size, (batch_size, seq_len)).to(device),
        "values_2": torch.randn(batch_size, seq_len).to(device),
        "panel_2": panel.to(device),
        "panel_name_1": "test_panel_1",
        "panel_name_2": "test_panel_2",
        "seq_length_1": [seq_len] * batch_size,
        "seq_length_2": [seq_len] * batch_size,
        "species": ["hsapiens"] * batch_size,
    }


# --------------------------------------------------------------------------------------
# Wiring tests (no CUDA / mamba_ssm required)
# --------------------------------------------------------------------------------------


def test_default_encoder_is_transformer():
    """Omitting encoder_type keeps the original transformer backbone (no behaviour change)."""
    config = _model_config()
    config.pop("encoder_type")  # simulate an old config without the key
    model = _build_model(config)
    assert type(model.transformer_encoder).__name__ == "TransformerEncoder"
    assert model.flash_attention is False  # config requested flash_attention=False


@pytest.mark.parametrize("encoder_type", MAMBA_VARIANTS)
def test_mamba_encoder_is_selected_and_disables_flash(encoder_type):
    """Selecting a mamba variant builds a MambaEncoder and forces the padded (non-flash) path."""
    if not MAMBA_AVAILABLE:
        pytest.skip("mamba_ssm is not installed")
    config = _model_config(encoder_type=encoder_type)
    config["flash_attention"] = True  # should be overridden to False for mamba
    model = _build_model(config)
    assert type(model.transformer_encoder).__name__ == "MambaEncoder"
    assert model.flash_attention is False
    assert len(model.transformer_encoder.layers) == config["nlayers"]
    # Bidirectional layers carry a reverse mixer.
    assert model.transformer_encoder.layers[0].mamba_bwd is not None


# --------------------------------------------------------------------------------------
# Forward / backward smoke tests (CUDA + mamba_ssm)
# --------------------------------------------------------------------------------------


@requires_mamba_gpu
@pytest.mark.parametrize("encoder_type", MAMBA_VARIANTS)
def test_mamba_predict_step_shapes(encoder_type):
    device = torch.device("cuda")
    config = _model_config(encoder_type=encoder_type)
    model = _build_model(config, device=device)
    model.eval()

    batch_size, seq_len = 6, 20
    batch = {
        "tokens": torch.randint(2, 64, (batch_size, seq_len)).to(device),
        "values": torch.randn(batch_size, seq_len).to(device),
        "seq_lengths": [seq_len] * batch_size,
        "species": ["hsapiens"] * batch_size,
    }

    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        out = model.predict_step(batch, batch_idx=0)

    assert out["cls_cell_emb"].shape == (batch_size, config["dim_model"])
    assert torch.isfinite(out["cls_cell_emb"]).all()


@requires_mamba_gpu
@pytest.mark.parametrize("encoder_type", MAMBA_VARIANTS)
def test_mamba_training_step_and_backward(encoder_type):
    device = torch.device("cuda")
    config = _model_config(encoder_type=encoder_type)
    model = _build_model(config, device=device)
    model.train()
    optimizer = ContrastiveModel.configure_optimizers(model)

    batch = _make_paired_batch(device)

    with patch.object(model, "log_metric"):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss = model.training_step(batch, batch_idx=0)

    assert isinstance(loss, torch.Tensor)
    assert loss.requires_grad
    assert torch.isfinite(loss)

    optimizer.zero_grad()
    loss.backward()

    # Gradients must reach the Mamba mixer parameters.
    first_layer = model.transformer_encoder.layers[0]
    grads = [p.grad for p in first_layer.mamba_fwd.parameters() if p.grad is not None]
    assert grads, "no gradients reached the forward Mamba mixer"
    assert any(torch.any(g != 0) for g in grads)

    optimizer.step()
