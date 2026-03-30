"""Multi-model registry backed by BentoML's model store.

Manages loading, scoring, and hot-swapping of multiple fraud detection models.
Supports champion/challenger pattern and ensemble scoring.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

try:
    import bentoml
    import bentoml.sklearn

    BENTOML_AVAILABLE = True
except ImportError:
    BENTOML_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class LoadedModel:
    name: str
    tag: str
    pipeline: Any
    metadata: dict
    is_champion: bool = False
    loaded_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class ModelRegistry:
    """Thread-safe multi-model registry with BentoML store integration."""

    def __init__(self) -> None:
        self._models: dict[str, LoadedModel] = {}
        self._champion_name: str | None = None
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Model store operations
    # ------------------------------------------------------------------

    def save_model(
        self,
        name: str,
        pipeline: Any,
        framework: str = "sklearn",
        metrics: dict | None = None,
    ) -> str:
        """Save a pipeline to BentoML's model store. Returns the tag string."""
        if not BENTOML_AVAILABLE:
            raise RuntimeError("BentoML is not installed")

        tag = bentoml.sklearn.save_model(
            "sentinel-fraud",
            pipeline,
            labels={"framework": framework, "name": name},
            metadata=metrics or {},
        )
        logger.info("Saved model %s to BentoML store: %s", name, tag)
        return str(tag)

    def load_model(self, tag: str, as_champion: bool = False) -> str:
        """Load a model from BentoML store into memory. Returns the model name."""
        if not BENTOML_AVAILABLE:
            raise RuntimeError("BentoML is not installed")

        model_ref = bentoml.models.get(tag)
        pipeline = bentoml.sklearn.load_model(model_ref)
        labels = model_ref.info.labels
        metadata = dict(model_ref.info.metadata)
        name = labels.get("name", str(model_ref.tag.version))

        loaded = LoadedModel(
            name=name,
            tag=str(model_ref.tag),
            pipeline=pipeline,
            metadata=metadata,
            is_champion=as_champion,
        )

        with self._lock:
            if as_champion:
                # Demote previous champion
                if self._champion_name and self._champion_name in self._models:
                    self._models[self._champion_name].is_champion = False
                self._champion_name = name
            self._models[name] = loaded

        logger.info("Loaded model %s (tag=%s, champion=%s)", name, tag, as_champion)
        return name

    def load_from_store(self, prefix: str = "sentinel-fraud") -> int:
        """Auto-load all models matching prefix from BentoML store.

        Returns the number of models loaded.
        """
        if not BENTOML_AVAILABLE:
            return 0

        loaded_count = 0
        try:
            model_list = bentoml.models.list(prefix)
        except Exception:
            logger.warning("Could not list BentoML models with prefix %s", prefix)
            return 0

        # Track the best model by AUC-PR for auto-champion
        best_auc_pr = -1.0
        best_name = None

        for model_ref in model_list:
            try:
                tag_str = str(model_ref.tag)
                name = self.load_model(tag_str)
                loaded_count += 1

                auc_pr = self._models[name].metadata.get("auc_pr", 0)
                if auc_pr > best_auc_pr:
                    best_auc_pr = auc_pr
                    best_name = name
            except Exception:
                logger.warning("Failed to load model %s", model_ref.tag, exc_info=True)

        # Auto-promote best model as champion
        if best_name is not None:
            self.set_champion(best_name)

        logger.info("Auto-loaded %d models from BentoML store", loaded_count)
        return loaded_count

    def load_pipeline_directly(
        self,
        name: str,
        pipeline: Any,
        metadata: dict | None = None,
        as_champion: bool = False,
    ) -> None:
        """Load a pipeline directly into memory (bypasses BentoML store).

        Useful for legacy joblib fallback or testing.
        """
        loaded = LoadedModel(
            name=name,
            tag=f"direct:{name}",
            pipeline=pipeline,
            metadata=metadata or {},
            is_champion=as_champion,
        )

        with self._lock:
            if as_champion:
                if self._champion_name and self._champion_name in self._models:
                    self._models[self._champion_name].is_champion = False
                self._champion_name = name
            self._models[name] = loaded

    # ------------------------------------------------------------------
    # Model management
    # ------------------------------------------------------------------

    def remove_model(self, name: str) -> bool:
        """Unload a model from memory. Returns True if removed."""
        with self._lock:
            if name not in self._models:
                return False
            del self._models[name]
            if self._champion_name == name:
                self._champion_name = None
                # Promote next available model if any
                if self._models:
                    new_champ = next(iter(self._models))
                    self._models[new_champ].is_champion = True
                    self._champion_name = new_champ
            return True

    def set_champion(self, name: str) -> None:
        """Promote a loaded model to champion."""
        with self._lock:
            if name not in self._models:
                raise KeyError(f"Model '{name}' is not loaded")
            if self._champion_name and self._champion_name in self._models:
                self._models[self._champion_name].is_champion = False
            self._models[name].is_champion = True
            self._champion_name = name

    def has_models(self) -> bool:
        return len(self._models) > 0

    @property
    def champion_name(self) -> str | None:
        return self._champion_name

    def list_models(self) -> list[dict]:
        """Return info about all loaded models."""
        with self._lock:
            return [
                {
                    "name": m.name,
                    "tag": m.tag,
                    "is_champion": m.is_champion,
                    "metadata": m.metadata,
                    "loaded_at": m.loaded_at.isoformat(),
                }
                for m in self._models.values()
            ]

    def list_store(self) -> list[dict]:
        """List all models in BentoML store (not just loaded ones)."""
        if not BENTOML_AVAILABLE:
            return []
        try:
            return [
                {
                    "tag": str(m.tag),
                    "labels": dict(m.info.labels),
                    "metadata": dict(m.info.metadata),
                }
                for m in bentoml.models.list("sentinel-fraud")
            ]
        except Exception:
            return []

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def score_champion(self, features: pd.DataFrame) -> float:
        """Score using the champion model. Raises if no champion."""
        with self._lock:
            if self._champion_name is None or self._champion_name not in self._models:
                raise RuntimeError("No champion model loaded")
            pipeline = self._models[self._champion_name].pipeline
        return float(pipeline.predict_proba(features)[0, 1])

    def score_all(self, features: pd.DataFrame) -> dict[str, float]:
        """Score with every loaded model."""
        with self._lock:
            models_snapshot = list(self._models.items())
        return {
            name: float(m.pipeline.predict_proba(features)[0, 1])
            for name, m in models_snapshot
        }

    def score_ensemble(self, features: pd.DataFrame) -> float:
        """Average score across all loaded models."""
        scores = self.score_all(features)
        if not scores:
            raise RuntimeError("No models loaded for ensemble scoring")
        return float(np.mean(list(scores.values())))

    def get_champion_model_name(self) -> str | None:
        """Return the name of the current champion model."""
        return self._champion_name
