"""Mamba-based sequence encoder as a drop-in replacement for ``TransformerEncoder``.

This module provides a Mamba (state-space model) backbone that mirrors the public
interface of :class:`concept.modules.transformer.TransformerEncoder`, so it can be swapped
into :class:`concept.model.ContrastiveModel` without touching the surrounding pipeline.

Supported variants (selected via ``encoder_type``): ``"mamba"`` / ``"mamba1"`` (``Mamba``),
``"mamba2"`` (``Mamba2``) and ``"mamba3"`` (``Mamba3``) from the optional ``mamba_ssm`` package.

The model reads the cell embedding from the CLS token at sequence position 0. A plain causal
Mamba scan would leave that position unable to see any downstream genes, so each layer runs a
**bidirectional** mixer (a forward scan plus a reverse scan, summed); the reverse scan lets the
CLS token aggregate the whole cell. This keeps CLS at position 0 and requires no change to the
rest of the model. ``mamba_ssm`` is imported lazily so the package keeps importing without it.
"""

import importlib
import importlib.util
import inspect
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# Detected without importing mamba_ssm (which needs CUDA kernels), mirroring how flash_attn
# availability is probed in concept.model.
MAMBA_AVAILABLE = importlib.util.find_spec("mamba_ssm") is not None

# Map encoder_type -> (module path, class name) inside mamba_ssm.
_MAMBA_CLASSES = {
    "mamba": ("mamba_ssm.modules.mamba_simple", "Mamba"),
    "mamba1": ("mamba_ssm.modules.mamba_simple", "Mamba"),
    "mamba2": ("mamba_ssm.modules.mamba2", "Mamba2"),
    "mamba3": ("mamba_ssm.modules.mamba3", "Mamba3"),
}


def _resolve_mamba_cls(encoder_type: str):
    """Lazily import and return the requested Mamba class."""
    key = encoder_type.lower()
    if key not in _MAMBA_CLASSES:
        raise ValueError(
            f"Unknown mamba encoder_type {encoder_type!r}; expected one of "
            f"{sorted(set(_MAMBA_CLASSES))}."
        )
    if not MAMBA_AVAILABLE:
        raise ImportError(
            f"encoder_type={encoder_type!r} requires the optional 'mamba_ssm' package "
            "(plus 'causal_conv1d'), which is not installed."
        )
    module_path, cls_name = _MAMBA_CLASSES[key]
    module = importlib.import_module(module_path)
    return getattr(module, cls_name)


def _get_activation_fn(activation: str):
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}")


class BiMambaEncoderLayer(nn.Module):
    """Pre/post-norm residual block with a (bi)directional Mamba mixer and an FFN.

    Structurally parallel to :class:`concept.modules.flash_attention_layer.FlashTransformerEncoderLayer`
    (same feed-forward sub-block, norms, dropouts and ``norm_scheme``); only the self-attention is
    replaced by a Mamba mixer.
    """

    def __init__(
        self,
        d_model: int,
        dim_feedforward: int,
        mamba_cls,
        mamba_kwargs: Optional[dict] = None,
        dropout: float = 0.1,
        activation: str = "gelu",
        layer_norm_eps: float = 1e-5,
        norm_scheme: str = "pre",
        bidirectional: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.bidirectional = bidirectional

        # Only forward kwargs the chosen Mamba class actually accepts, so a shared config block
        # (e.g. containing d_conv) does not break Mamba3, whose signature differs.
        mamba_kwargs = dict(mamba_kwargs or {})
        valid = set(inspect.signature(mamba_cls.__init__).parameters)
        filtered = {k: v for k, v in mamba_kwargs.items() if k in valid}
        # Mamba modules are created on the default device and moved with the parent .to(...).
        self.mamba_fwd = mamba_cls(d_model, **filtered)
        self.mamba_bwd = mamba_cls(d_model, **filtered) if bidirectional else None

        # Feed-forward network (identical layout to the transformer layer).
        self.linear1 = nn.Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, **factory_kwargs)

        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.norm_scheme = norm_scheme
        if self.norm_scheme not in ("pre", "post"):
            raise ValueError(f"norm_scheme should be pre or post, not {norm_scheme}")

    def _run_mamba(self, mamba: nn.Module, x: Tensor) -> Tensor:
        """Run a single Mamba scan in the parameter dtype with autocast disabled.

        The SSM recurrence is numerically sensitive and its CUDA kernels expect the activations
        and parameters to share a dtype. Running the mixer in the module's parameter dtype (fp32
        under standard AMP training) avoids autocast dtype mismatches and keeps the scan stable;
        the surrounding FFN/projections still benefit from autocast. The result is cast back to
        the residual-stream dtype.
        """
        param_dtype = next(mamba.parameters()).dtype
        device_type = x.device.type
        with torch.autocast(device_type=device_type, enabled=False):
            return mamba(x.to(param_dtype)).to(x.dtype)

    def _mix(self, x: Tensor, key_padding_mask: Optional[Tensor]) -> Tensor:
        # Zero padded positions so they do not leak into the scan/conv.
        if key_padding_mask is not None:
            x = x.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)

        out = self._run_mamba(self.mamba_fwd, x)
        if self.mamba_bwd is not None:
            # Reverse along the sequence; with right-padding the leading zeros contribute ~nothing
            # and the reverse scan reaches CLS (position 0) last, having integrated every token.
            x_rev = torch.flip(x, dims=[1])
            out_rev = self._run_mamba(self.mamba_bwd, x_rev)
            out = out + torch.flip(out_rev, dims=[1])
        return out

    def forward(
        self,
        src: Tensor,
        src_mask: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
        cu_seqlens: Optional[Tensor] = None,
        max_seqlen: Optional[Tensor] = None,
        **kwargs,
    ) -> Tensor:
        if src_mask is not None:
            raise ValueError("BiMambaEncoderLayer does not support src_mask")

        if self.norm_scheme == "pre":
            mix_src = self.norm1(src)
            src2 = self._mix(mix_src, key_padding_mask)
            src = src + self.dropout1(src2)
            src2 = self.linear2(self.dropout(self.activation(self.linear1(self.norm2(src)))))
            src = src + self.dropout2(src2)
        else:
            src2 = self._mix(src, key_padding_mask)
            src = self.norm1(src + self.dropout1(src2))
            src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
            src = self.norm2(src + self.dropout2(src2))
        return src


class MambaEncoder(nn.Module):
    """Stack of :class:`BiMambaEncoderLayer` exposing the ``TransformerEncoder`` interface.

    Accepts (and ignores) the flash-path ``cu_seqlens`` / ``max_seqlen`` arguments so it can be
    called exactly like ``TransformerEncoder``; Mamba uses the padded + ``key_padding_mask`` path.
    """

    def __init__(
        self,
        d_model: int,
        dim_feedforward: int,
        nlayers: int,
        encoder_type: str = "mamba",
        mamba_kwargs: Optional[dict] = None,
        dropout: float = 0.1,
        activation: str = "gelu",
        norm_scheme: str = "pre",
        layer_norm_eps: float = 1e-5,
        norm: Optional[nn.Module] = None,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        mamba_kwargs = dict(mamba_kwargs or {})
        bidirectional = bool(mamba_kwargs.pop("bidirectional", True))
        mamba_cls = _resolve_mamba_cls(encoder_type)

        self.layers = nn.ModuleList(
            [
                BiMambaEncoderLayer(
                    d_model=d_model,
                    dim_feedforward=dim_feedforward,
                    mamba_cls=mamba_cls,
                    mamba_kwargs=mamba_kwargs,
                    dropout=dropout,
                    activation=activation,
                    layer_norm_eps=layer_norm_eps,
                    norm_scheme=norm_scheme,
                    bidirectional=bidirectional,
                    device=device,
                    dtype=dtype,
                )
                for _ in range(nlayers)
            ]
        )
        self.num_layers = nlayers
        self.norm = norm

    def forward(
        self,
        src: Tensor,
        mask: Optional[Tensor] = None,
        cu_seqlens: Optional[Tensor] = None,
        max_seqlen: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
        is_causal: bool = False,
        **kwargs,
    ) -> Tensor:
        output = src
        for mod in self.layers:
            output = mod(output, mask, key_padding_mask=key_padding_mask)
        if self.norm is not None:
            output = self.norm(output)
        return output
