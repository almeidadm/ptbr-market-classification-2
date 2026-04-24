"""Testes para `src.modelos.logreg`."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import GRID_LOGREG_C, GRID_LOGREG_CLASS_WEIGHT
from src.modelos.logreg import (
    construir_pipeline,
    grid_logreg,
    predizer_logreg,
    treinar_logreg,
)


def _corpus_sintetico(n: int = 200) -> tuple[pd.Series, pd.Series]:
    rng = np.random.default_rng(42)
    classes = rng.choice(["mercado", "poder", "outros"], size=n, p=[0.2, 0.3, 0.5])
    tokens_por_classe = {
        "mercado": ["bolsa empresa dólar inflação pib", "mercado financeiro"],
        "poder": ["congresso senado governo presidente lei", "político partido"],
        "outros": ["geral cotidiano acontecimento várias notícias", "outra coisa"],
    }
    textos = [
        f"{tokens_por_classe[c][i % 2]} exemplo {i}" for i, c in enumerate(classes)
    ]
    return pd.Series(textos), pd.Series(classes)


def test_grid_tem_seis_configuracoes() -> None:
    grid = grid_logreg()
    assert len(grid) == len(GRID_LOGREG_C) * len(GRID_LOGREG_CLASS_WEIGHT)
    assert all("C" in cfg and "class_weight" in cfg for cfg in grid)


def test_construir_pipeline_retorna_tfidf_mais_clf() -> None:
    pipeline = construir_pipeline({"C": 1.0, "class_weight": None})
    assert list(pipeline.named_steps.keys()) == ["tfidf", "clf"]


def test_treinar_e_predizer_smoke() -> None:
    x, y = _corpus_sintetico()
    pipeline = treinar_logreg(x, y, {"C": 1.0, "class_weight": None})
    y_pred, probs, classes = predizer_logreg(pipeline, x.iloc[:20])

    assert len(y_pred) == 20
    assert probs.shape == (20, len(classes))
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-6)
    assert set(classes) == {"mercado", "poder", "outros"}


def test_determinismo_sob_mesma_seed() -> None:
    x, y = _corpus_sintetico()
    p1 = treinar_logreg(x, y, {"C": 1.0, "class_weight": None})
    p2 = treinar_logreg(x, y, {"C": 1.0, "class_weight": None})
    _, probs1, _ = predizer_logreg(p1, x.iloc[:20])
    _, probs2, _ = predizer_logreg(p2, x.iloc[:20])
    np.testing.assert_array_almost_equal(probs1, probs2)
