"""
Família TF-IDF + SVM linear (ADR 009 §D.1, §D.3).

Leitura B de Q8 (2026-04-24): `LinearSVC` puro, sem `CalibratedClassifierCV`,
em ambos os protocolos (rolling e k-fold). PR-AUC é calculada sobre
`decision_function` em vez de probabilidades calibradas; a ADR 009 §D.3
precisa ser revisada formalmente para refletir essa mudança (task 7).

Grid de 6 configurações (`C × class_weight`).
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from src.config import (
    GRID_SVM_C,
    GRID_SVM_CLASS_WEIGHT,
    SVM_FIXOS,
    TFIDF_CONFIG,
)


def grid_svm() -> list[dict]:
    return [
        {"C": c, "class_weight": cw}
        for c, cw in itertools.product(GRID_SVM_C, GRID_SVM_CLASS_WEIGHT)
    ]


def construir_pipeline(hp: dict) -> Pipeline:
    return Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(**TFIDF_CONFIG)),
            ("clf", LinearSVC(**SVM_FIXOS, **hp)),
        ]
    )


def treinar_svm(x: pd.Series, y: pd.Series, hp: dict) -> Pipeline:
    pipeline = construir_pipeline(hp)
    pipeline.fit(x, y)
    return pipeline


def predizer_svm(
    pipeline: Pipeline, x: pd.Series
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Retorna `(y_pred, scores, classes)`.

    `scores` é `decision_function(x)` com forma `(n_amostras, n_classes)`
    no caso multiclasse (one-vs-rest internamente em `LinearSVC`) ou
    `(n_amostras,)` no caso binário — convertido para 2D com stack
    `[-score, score]` para uniformizar com o caso multiclasse quando o
    regime é binário.
    """
    y_pred = pipeline.predict(x)
    scores = pipeline.decision_function(x)
    if scores.ndim == 1:
        scores = np.column_stack([-scores, scores])
    classes = list(pipeline.named_steps["clf"].classes_)
    return y_pred, scores, classes
