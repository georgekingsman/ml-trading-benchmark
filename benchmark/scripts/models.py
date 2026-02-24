"""
models.py — Unified model interface and 8+1 baseline implementations.

Every model exposes:
    .fit(X_train, y_train, X_val=None, y_val=None)
    .predict(X)  → np.ndarray of predicted forward returns

Models
------
Traditional (4):  LinearRegression, Ridge, LogisticRegression*, RandomForest, LightGBM
Deep (2):         MLP, LSTM
Strategy (2):     MomentumBaseline, MeanReversionBaseline

* LogisticRegression predicts direction probability; we convert to
  a "score" comparable with regression predictions.
"""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from typing import Any, Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ================================================================== #
#  Abstract base
# ================================================================== #

class BaseModel(ABC):
    name: str = "base"

    @abstractmethod
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> "BaseModel":
        ...

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return predicted forward return (float) for each row."""
        ...

    def __repr__(self):
        return f"<{self.name}>"


# ================================================================== #
#  Traditional ML
# ================================================================== #

class LinearRegressionModel(BaseModel):
    name = "LinearRegression"

    def __init__(self, **kw):
        from sklearn.linear_model import LinearRegression
        self.model = LinearRegression(**kw)

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        self.model.fit(X_train, y_train)
        return self

    def predict(self, X):
        return self.model.predict(X)


class RidgeModel(BaseModel):
    name = "Ridge"

    def __init__(self, alpha: float = 1.0, **kw):
        from sklearn.linear_model import Ridge
        self.model = Ridge(alpha=alpha, **kw)

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        self.model.fit(X_train, y_train)
        return self

    def predict(self, X):
        return self.model.predict(X)


class LogisticRegressionModel(BaseModel):
    """Predicts P(direction=up); output is treated as a signal score."""
    name = "LogisticRegression"

    def __init__(self, C: float = 1.0, **kw):
        from sklearn.linear_model import LogisticRegression
        self.model = LogisticRegression(C=C, max_iter=1000, **kw)

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        # Convert regression target to direction
        y_dir = (y_train > 0).astype(int)
        self.model.fit(X_train, y_dir)
        return self

    def predict(self, X):
        # Return probability of up direction as signal
        return self.model.predict_proba(X)[:, 1] - 0.5  # centered around 0


class RandomForestModel(BaseModel):
    name = "RandomForest"

    def __init__(self, n_estimators: int = 200, max_depth: int = 8,
                 random_state: int = 42, **kw):
        from sklearn.ensemble import RandomForestRegressor
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1,
            **kw,
        )

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        self.model.fit(X_train, y_train)
        return self

    def predict(self, X):
        return self.model.predict(X)


class LightGBMModel(BaseModel):
    name = "LightGBM"

    def __init__(self, n_estimators: int = 300, max_depth: int = 6,
                 learning_rate: float = 0.05, random_state: int = 42, **kw):
        import lightgbm as lgb
        self.model = lgb.LGBMRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=random_state,
            n_jobs=-1,
            verbose=-1,
            **kw,
        )

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        fit_params: dict[str, Any] = {}
        if X_val is not None and y_val is not None:
            fit_params["eval_set"] = [(X_val, y_val)]
            fit_params["callbacks"] = [
                __import__("lightgbm").early_stopping(50, verbose=False),
            ]
        self.model.fit(X_train, y_train, **fit_params)
        return self

    def predict(self, X):
        return self.model.predict(X)


# ================================================================== #
#  Deep Learning
# ================================================================== #

class MLPModel(BaseModel):
    name = "MLP"

    def __init__(self, hidden_dims: list[int] | None = None, epochs: int = 50,
                 lr: float = 1e-3, batch_size: int = 256, **kw: Any):
        self.hidden_dims = hidden_dims or [128, 64]
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.model: Any = None
        self.device: Any = None

    @staticmethod
    def _get_device():
        import torch
        if torch.cuda.is_available():
            return torch.device("cuda")
        # MPS (Apple Metal) can hang during training — fall back to CPU
        return torch.device("cpu")

    def _build(self, input_dim: int):
        import torch
        import torch.nn as nn
        self.device = self._get_device()
        layers = []
        prev = input_dim
        for h in self.hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(0.2)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.model = nn.Sequential(*layers).to(self.device)

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
        from tqdm import tqdm

        self._build(X_train.shape[1])
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()

        Xt = torch.tensor(X_train, dtype=torch.float32)
        yt = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
        ds = TensorDataset(Xt, yt)
        dl = DataLoader(ds, batch_size=self.batch_size, shuffle=True)

        self.model.train()
        pbar = tqdm(range(self.epochs), desc=f"MLP [{self.device}]", leave=False)
        for ep in pbar:
            epoch_loss = 0.0
            for xb, yb in dl:
                xb, yb = xb.to(self.device), yb.to(self.device)
                opt.zero_grad()
                loss = loss_fn(self.model(xb), yb)
                loss.backward()
                opt.step()
                epoch_loss += loss.item()
            pbar.set_postfix(loss=f"{epoch_loss / len(dl):.6f}")
        return self

    def predict(self, X):
        import torch
        self.model.eval()
        with torch.no_grad():
            Xt = torch.tensor(X, dtype=torch.float32).to(self.device)
            return self.model(Xt).cpu().numpy().flatten()


class LSTMModel(BaseModel):
    name = "LSTM"

    def __init__(self, hidden_dim: int = 64, num_layers: int = 2,
                 seq_len: int = 20, epochs: int = 30, lr: float = 1e-3,
                 batch_size: int = 256, **kw):
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.seq_len = seq_len
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.model: Any = None
        self.device: Any = None

    def _build(self, input_dim: int):
        import torch
        import torch.nn as nn

        class _LSTM(nn.Module):
            def __init__(self, input_dim, hidden_dim, num_layers):  # noqa: N805
                super().__init__()
                self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers,
                                    batch_first=True, dropout=0.2)
                self.fc = nn.Linear(hidden_dim, 1)

            def forward(self, x):  # noqa: N805
                # x: (batch, seq_len, features)
                out, _ = self.lstm(x)
                return self.fc(out[:, -1, :])

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            # MPS (Apple Metal) can hang during training — fall back to CPU
            self.device = torch.device("cpu")
        self.model = _LSTM(input_dim, self.hidden_dim, self.num_layers).to(self.device)

    def _make_sequences(self, X: np.ndarray) -> np.ndarray:
        """Reshape flat (N, F) → (N-seq+1, seq_len, F) using rolling windows."""
        seqs = []
        for i in range(self.seq_len - 1, len(X)):
            seqs.append(X[i - self.seq_len + 1 : i + 1])
        return np.array(seqs)

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
        from tqdm import tqdm

        X_seq = self._make_sequences(X_train)
        y_seq = y_train[self.seq_len - 1 :]

        self._build(X_train.shape[1])
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()

        Xt = torch.tensor(X_seq, dtype=torch.float32)
        yt = torch.tensor(y_seq, dtype=torch.float32).unsqueeze(1)
        ds = TensorDataset(Xt, yt)
        dl = DataLoader(ds, batch_size=self.batch_size, shuffle=True)

        self.model.train()
        pbar = tqdm(range(self.epochs), desc=f"LSTM [{self.device}]", leave=False)
        for ep in pbar:
            epoch_loss = 0.0
            for xb, yb in dl:
                xb, yb = xb.to(self.device), yb.to(self.device)
                opt.zero_grad()
                loss = loss_fn(self.model(xb), yb)
                loss.backward()
                opt.step()
                epoch_loss += loss.item()
            pbar.set_postfix(loss=f"{epoch_loss / len(dl):.6f}")
        return self

    def predict(self, X):
        import torch
        self.model.eval()
        X_seq = self._make_sequences(X)
        with torch.no_grad():
            Xt = torch.tensor(X_seq, dtype=torch.float32).to(self.device)
            preds = self.model(Xt).cpu().numpy().flatten()
        # Pad beginning with NaN (no prediction for first seq_len-1 rows)
        return np.concatenate([np.full(self.seq_len - 1, np.nan), preds])


# ================================================================== #
#  Simple strategy baselines
# ================================================================== #

class MomentumBaseline(BaseModel):
    """Signal = past N-day return (no training needed)."""
    name = "MomentumBaseline"

    def __init__(self, lookback: int = 20, **kw):
        self.lookback = lookback
        # Expects 'return_{lookback}d' to be in the feature columns
        self._col_idx: Optional[int] = None

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        # No training; we just use past returns as signal
        return self

    def predict(self, X):
        # Momentum signal: sum of daily returns in window
        # We use the return_20d column directly (column index 2 in FEATURE_COLS)
        # return_1d=0, return_5d=1, return_20d=2
        return X[:, 2]  # return_20d


class MeanReversionBaseline(BaseModel):
    """Signal = negative of past N-day return (contrarian)."""
    name = "MeanReversionBaseline"

    def __init__(self, lookback: int = 20, **kw):
        self.lookback = lookback

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        return self

    def predict(self, X):
        return -X[:, 2]  # negative return_20d


# ================================================================== #
#  Model registry
# ================================================================== #

MODEL_REGISTRY: dict[str, type[BaseModel]] = {
    "LinearRegression":      LinearRegressionModel,
    "Ridge":                 RidgeModel,
    "LogisticRegression":    LogisticRegressionModel,
    "RandomForest":          RandomForestModel,
    "LightGBM":             LightGBMModel,
    "MLP":                  MLPModel,
    "LSTM":                 LSTMModel,
    "MomentumBaseline":     MomentumBaseline,
    "MeanReversionBaseline": MeanReversionBaseline,
}


def build_model(name: str, params: dict | None = None) -> BaseModel:
    """Instantiate a model by name from the registry."""
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {name}. Available: {list(MODEL_REGISTRY.keys())}")
    params = params or {}
    return MODEL_REGISTRY[name](**params)
