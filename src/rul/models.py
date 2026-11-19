"""Sequence models for Remaining-Useful-Life regression.

Three architectures are benchmarked head-to-head under an identical training budget:

* ``LSTMRegressor``   - the classic recurrent baseline used in most RUL papers.
* ``TCNRegressor``    - temporal convolutional network (dilated causal convolutions).
* ``RULTransformer``  - encoder-only transformer with learned positional embeddings,
                        a [CLS] token and attention pooling. Monte-Carlo dropout is
                        supported for epistemic-uncertainty estimates.
"""
from __future__ import annotations

import torch
from torch import nn


class LSTMRegressor(nn.Module):
    def __init__(self, n_features: int, hidden: int = 64, layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, num_layers=layers, batch_first=True, dropout=dropout)
        self.head = nn.Sequential(nn.Linear(hidden, 32), nn.ReLU(), nn.Dropout(dropout), nn.Linear(32, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.head(out[:, -1]).squeeze(-1)


class _CausalBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int, k: int, dilation: int, dropout: float):
        super().__init__()
        self.pad = (k - 1) * dilation
        self.conv1 = nn.Conv1d(c_in, c_out, k, dilation=dilation)
        self.conv2 = nn.Conv1d(c_out, c_out, k, dilation=dilation)
        self.drop = nn.Dropout(dropout)
        self.res = nn.Conv1d(c_in, c_out, 1) if c_in != c_out else nn.Identity()

    def forward(self, x):
        y = nn.functional.pad(x, (self.pad, 0))
        y = self.drop(torch.relu(self.conv1(y)))
        y = nn.functional.pad(y, (self.pad, 0))
        y = self.drop(torch.relu(self.conv2(y)))
        return torch.relu(y + self.res(x))


class TCNRegressor(nn.Module):
    def __init__(self, n_features: int, channels: int = 64, levels: int = 4, k: int = 3, dropout: float = 0.2):
        super().__init__()
        blocks, c = [], n_features
        for i in range(levels):
            blocks.append(_CausalBlock(c, channels, k, 2 ** i, dropout))
            c = channels
        self.tcn = nn.Sequential(*blocks)
        self.head = nn.Sequential(nn.Linear(channels, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x):
        y = self.tcn(x.transpose(1, 2))           # (B, C, T)
        return self.head(y[:, :, -1]).squeeze(-1)


class RULTransformer(nn.Module):
    def __init__(self, n_features: int, window: int = 30, d_model: int = 64, heads: int = 4,
                 layers: int = 3, ff: int = 128, dropout: float = 0.15):
        super().__init__()
        self.embed = nn.Sequential(nn.Linear(n_features, d_model), nn.LayerNorm(d_model))
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos = nn.Parameter(torch.randn(1, window + 1, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(d_model, heads, ff, dropout, batch_first=True,
                                           norm_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(nn.Linear(d_model, 32), nn.GELU(), nn.Dropout(dropout), nn.Linear(32, 1))

    def forward(self, x):
        b, t, _ = x.shape
        h = self.embed(x)
        h = torch.cat([self.cls.expand(b, -1, -1), h], dim=1) + self.pos[:, : t + 1]
        h = self.norm(self.encoder(h))
        return self.head(h[:, 0]).squeeze(-1)


MODELS = {"lstm": LSTMRegressor, "tcn": TCNRegressor, "transformer": RULTransformer}


def build_model(name: str, n_features: int, window: int) -> nn.Module:
    if name == "transformer":
        return RULTransformer(n_features, window=window)
    return MODELS[name](n_features)


@torch.no_grad()
def mc_dropout_predict(model: nn.Module, x: torch.Tensor, passes: int = 30) -> tuple[torch.Tensor, torch.Tensor]:
    """Monte-Carlo dropout: keep dropout active at inference to sample the predictive distribution."""
    model.train()
    preds = torch.stack([model(x) for _ in range(passes)]) * 125.0     # models regress RUL / 125
    model.eval()
    return preds.mean(0), preds.std(0)
