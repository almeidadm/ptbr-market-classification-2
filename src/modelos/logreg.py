"""
Família TF-IDF + Regressão Logística (ADR 009 §D.1, §D.3).

Grid de 6 configurações (`C × class_weight`). Fixos conforme ADR 009:
`solver="liblinear"`, `max_iter=2000`, `random_state=SEED`.
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.config import (
    GRID_LOGREG_C,
    GRID_LOGREG_CLASS_WEIGHT,
    LOGREG_FIXOS,
    TFIDF_CONFIG,
)


def grid_logreg() -> list[dict]:
    """Enumera o grid na ordem canônica `(C, class_weight)`."""
    return [
        {"C": c, "class_weight": cw}
        for c, cw in itertools.product(GRID_LOGREG_C, GRID_LOGREG_CLASS_WEIGHT)
    ]


def construir_pipeline(hp: dict) -> Pipeline:
    return Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(**TFIDF_CONFIG)),
            ("clf", LogisticRegression(**LOGREG_FIXOS, **hp)),
        ]
    )


def treinar_logreg(x: pd.Series, y: pd.Series, hp: dict) -> Pipeline:
    pipeline = construir_pipeline(hp)
    pipeline.fit(x, y)
    return pipeline


def predizer_logreg(
    pipeline: Pipeline, x: pd.Series
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Retorna `(y_pred, probs, classes)`.

    `probs` tem forma `(n_amostras, n_classes)` e `classes` reflete a
    ordem usada pelo `LogisticRegression` (alfabética em sklearn).
    """
    y_pred = pipeline.predict(x)
    probs = pipeline.predict_proba(x)
    classes = list(pipeline.named_steps["clf"].classes_)
    return y_pred, probs, classes
