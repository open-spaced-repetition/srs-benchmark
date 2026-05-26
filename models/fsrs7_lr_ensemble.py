import copy
import numpy as np
import torch
import pandas as pd
from typing import Any, Optional, Dict

from config import Config
from models.base import BaseModel
from models.fsrs_v7 import FSRS7
from models.logistic_regression import LogisticRegression


class FSRS7LREnsemble(BaseModel):
    """Ensemble of FSRS-7 and Logistic Regression via logit-space blending.

    Blending weights are optimized on out-of-sample predictions from a
    temporal holdout (last 30% of training data) using
    ``scipy.optimize.minimize_scalar`` with an L2 prior centered at 0.4
    to regularize toward equal weighting on small datasets. Final base
    models are retrained on the full training set. Weights are clamped to
    [0.1, 0.9] to ensure neither model is fully ignored.

    Separate weights are learned for same-day and non-same-day reviews.
    """

    Trainer: Any = None
    PRIOR = np.array([0.4, 0.4])

    def __init__(self, config: Config, state_dict: Optional[Dict[str, Any]] = None):
        super().__init__(config)

        self.fsrs_config = copy.deepcopy(config)
        self.fsrs_config.model_name = "FSRS-7"

        self.lr_config = copy.deepcopy(config)
        self.lr_config.model_name = "LogisticRegression"

        if state_dict is not None:
            self.weights = state_dict
        else:
            self.weights = {"fsrs_w": None, "lr_state": None}

    # ── helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _to_logits(arr):
        a = np.clip(np.asarray(arr, dtype=np.float64), 1e-7, 1 - 1e-7)
        return np.log(a / (1.0 - a))

    @staticmethod
    def _sigmoid(x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))

    W_MIN, W_MAX = 0.1, 0.9

    @classmethod
    def _fit_weight(cls, logit_fsrs, logit_lr, y):
        """Find the blending weight w that minimizes BCE + L2 prior on a holdout."""
        from scipy.optimize import minimize_scalar

        prior_w = 0.4
        n = len(y)
        reg = max(1.0 / max(n, 1), 1e-3)

        def loss_fn(w):
            logit = w * logit_fsrs + (1.0 - w) * logit_lr
            p = cls._sigmoid(logit)
            p = np.clip(p, 1e-7, 1 - 1e-7)
            bce = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
            return bce + reg * (w - prior_w) ** 2

        result = minimize_scalar(
            loss_fn, bounds=(cls.W_MIN, cls.W_MAX), method="bounded"
        )
        return float(result.x)

    # ── logit blending ─────────────────────────────────────────────────

    def _ensemble_logits(self, logit_fsrs, logit_lr, is_same_day):
        w_nsd = self.weights.get("w_fsrs_nsd", self.PRIOR[0])
        w_sd = self.weights.get("w_fsrs_sd", self.PRIOR[1])
        w = np.where(is_same_day, w_sd, w_nsd)
        return w * logit_fsrs + (1.0 - w) * logit_lr

    # ── training ─────────────────────────────────────────────────────────

    def _train_fsrs(self, df: pd.DataFrame):
        if self.Trainer is None:
            raise RuntimeError("Trainer must be registered before optimizing the ensemble")
        fsrs_model = FSRS7(config=self.fsrs_config)
        trainer = self.Trainer(
            model=fsrs_model,
            train_set=df,
            test_set=None,
            batch_size=self.fsrs_config.batch_size,
        )
        return trainer.train()

    def _train_lr(self, df: pd.DataFrame):
        return LogisticRegression(config=self.lr_config).optimize(df.copy())

    def _fit_blending_weights(self, train_df: pd.DataFrame, val_df: pd.DataFrame):
        from utils import Collection

        fsrs_w = self._train_fsrs(train_df)
        lr_state = self._train_lr(train_df)

        fsrs_model = FSRS7(config=self.fsrs_config, w=fsrs_w)
        fsrs_ret, _, _ = Collection(fsrs_model, self.fsrs_config).batch_predict(val_df)
        logit_fsrs = self._to_logits(np.array(fsrs_ret))

        lr_model = LogisticRegression(config=self.lr_config, state_dict=lr_state)
        logit_lr = self._to_logits(lr_model.predict(val_df).cpu().numpy())

        y_val = val_df["y"].values.astype(np.float64)
        is_same_day = (val_df["delta_t_int"].values == 0).astype(bool)
        nsd_mask = ~is_same_day

        w_nsd = self.PRIOR[0]
        if np.sum(nsd_mask) >= 10:
            w_nsd = self._fit_weight(
                logit_fsrs[nsd_mask], logit_lr[nsd_mask], y_val[nsd_mask]
            )

        w_sd = self.PRIOR[1]
        if np.sum(is_same_day) >= 10:
            w_sd = self._fit_weight(
                logit_fsrs[is_same_day], logit_lr[is_same_day], y_val[is_same_day]
            )

        return w_nsd, w_sd

    def optimize(self, df: pd.DataFrame) -> Dict[str, Any]:
        n = len(df)
        val_start = max(1, int(n * 0.7))
        w_nsd, w_sd = self.PRIOR

        if n - val_start >= 10:
            train_df = df.iloc[:val_start].copy()
            val_df = df.iloc[val_start:].copy()
            w_nsd, w_sd = self._fit_blending_weights(train_df, val_df)

        fsrs_w = self._train_fsrs(df)
        lr_state = self._train_lr(df)

        self.weights = {
            "fsrs_w": fsrs_w,
            "lr_state": lr_state,
            "w_fsrs_nsd": w_nsd,
            "w_fsrs_sd": w_sd,
        }
        self._fsrs_model = None
        self._lr_model = None
        return self.weights

    # ── inference ────────────────────────────────────────────────────────

    def _get_sub_models(self):
        if not hasattr(self, "_fsrs_model") or self._fsrs_model is None:
            self._fsrs_model = FSRS7(
                config=self.fsrs_config, w=self.weights.get("fsrs_w")
            )
            self._lr_model = LogisticRegression(
                config=self.lr_config, state_dict=self.weights.get("lr_state")
            )
        return self._fsrs_model, self._lr_model

    @torch.inference_mode()
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        fsrs_model, lr_model = self._get_sub_models()
        from utils import Collection

        fsrs_ret, _, _ = Collection(fsrs_model, self.fsrs_config).batch_predict(df)
        lr_ret = lr_model.predict(df).cpu().numpy()

        logit_fsrs = self._to_logits(np.array(fsrs_ret))
        logit_lr = self._to_logits(np.array(lr_ret))
        is_same_day = df["delta_t_int"].values == 0

        logit_ens = self._ensemble_logits(logit_fsrs, logit_lr, is_same_day)
        preds = self._sigmoid(logit_ens)
        return np.clip(preds, 1e-7, 1 - 1e-7)

    # ── serialization ────────────────────────────────────────────────────

    def benchmark_state(self) -> Dict[str, Any]:
        return self.weights

    def log(self, x: Dict[str, Any]) -> None:
        params = []
        fsrs_w = self.weights.get("fsrs_w")
        if fsrs_w is not None:
            params.extend([round(p, 4) for p in fsrs_w])
        lr_state = self.weights.get("lr_state")
        if lr_state is not None:
            lr_model = LogisticRegression(config=self.lr_config, state_dict=lr_state)
            params.extend([round(p, 4) for p in lr_model.coefficients.tolist()])
        params.append(round(self.weights.get("w_fsrs_nsd", self.PRIOR[0]), 4))
        params.append(round(self.weights.get("w_fsrs_sd", self.PRIOR[1]), 4))
        x["parameters"] = params

    def load_state_dict(
        self, state_dict: Dict[str, Any], strict: bool = True, assign: bool = False
    ) -> Any:
        self.weights = state_dict
        self._fsrs_model = None
        self._lr_model = None
        return self
