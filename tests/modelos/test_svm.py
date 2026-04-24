"""Testes para `src.modelos.svm` (Leitura B de Q8: sem calibração)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import GRID_SVM_C, GRID_SVM_CLASS_WEIGHT
from src.modelos.svm import (
    construir_pipeline,
    grid_svm,
    predizer_svm,
    treinar_svm,
)


def _corpus() -> tuple[pd.Series, pd.Series]:
    rng = np.random.default_rng(7)
    classes = rng.choice(["mercado", "poder", "outros"], size=150, p=[0.2, 0.3, 0.5])
    vocab = {
        "mercado": "bolsa empresa dólar inflação pib",
        "poder": "congresso senado governo presidente lei",
        "outros": "cotidiano geral variado notícias diversas",
    }
    textos = [f"{vocab[c]} exemplo {i}" for i, c in enumerate(classes)]
    return pd.Series(textos), pd.Series(classes)


def test_grid_tem_seis_configuracoes() -> None:
    grid = grid_svm()
    assert len(grid) == len(GRID_SVM_C) * len(GRID_SVM_CLASS_WEIGHT)


def test_pipeline_sem_calibrador_apenas_linearsvc() -> None:
    pipeline = construir_pipeline({"C": 1.0, "class_weight": None})
    nomes = list(pipeline.named_steps.keys())
    assert nomes == ["tfidf", "clf"]
    assert type(pipeline.named_steps["clf"]).__name__ == "LinearSVC"


def test_decision_function_retornada_com_forma_2d_mesmo_multiclasse() -> None:
    x, y = _corpus()
    pipeline = treinar_svm(x, y, {"C": 1.0, "class_weight": None})
    y_pred, scores, classes = predizer_svm(pipeline, x.iloc[:30])

    assert len(y_pred) == 30
    assert scores.shape == (30, len(classes))


def test_decision_function_retornada_com_forma_2d_mesmo_binario() -> None:
    # LinearSVC com duas classes retorna decision_function 1-D; o wrapper
    # converte para (n, 2) para o runner poder tratar de forma uniforme.
    rng = np.random.default_rng(0)
    y_bin = pd.Series(rng.choice(["mercado", "nao-mercado"], size=100))
    x_bin = pd.Series(
        [
            "bolsa empresa" if c == "mercado" else "cotidiano variado"
            for c in y_bin
        ]
    )
    pipeline = treinar_svm(x_bin, y_bin, {"C": 1.0, "class_weight": None})
    _, scores, classes = predizer_svm(pipeline, x_bin.iloc[:10])
    assert scores.shape == (10, 2)
    assert len(classes) == 2


def test_determinismo_sob_mesma_seed() -> None:
    x, y = _corpus()
    p1 = treinar_svm(x, y, {"C": 1.0, "class_weight": None})
    p2 = treinar_svm(x, y, {"C": 1.0, "class_weight": None})
    _, s1, _ = predizer_svm(p1, x.iloc[:30])
    _, s2, _ = predizer_svm(p2, x.iloc[:30])
    np.testing.assert_array_almost_equal(s1, s2)
