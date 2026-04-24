"""Testes para `src.experimento.metricas`."""
from __future__ import annotations

import numpy as np

from src.experimento.metricas import (
    bootstrap_ic,
    calcular_metricas_fold,
    projetar_binario,
    resumir_cross_fold,
)


def test_projetar_binario_retorna_1_para_mercado_e_0_para_resto() -> None:
    y = np.array(["mercado", "poder", "outros", "mercado", "esporte"])
    resultado = projetar_binario(y)
    np.testing.assert_array_equal(resultado, [1, 0, 0, 1, 0])


def test_f1_binario_sobre_classes_multi_coincide_com_f1_binario_direto() -> None:
    classes = ["mercado", "outros", "poder"]
    y_true = np.array(["mercado", "mercado", "outros", "poder", "mercado"])
    y_pred = np.array(["mercado", "outros", "outros", "poder", "mercado"])
    scores = np.full((5, 3), 1 / 3)
    m = calcular_metricas_fold(y_true, y_pred, scores, classes)
    # TP=2 (mercado pred bem), FN=1 (mercado virou outros), FP=0.
    # Precisão=1, Recall=2/3, F1=0.8.
    assert abs(m["f1_binario_projetado"] - 0.8) < 1e-6


def test_pr_auc_bem_calibrado_quando_scores_rankeiam_positivos_acima() -> None:
    classes = ["mercado", "outros"]
    y_true = np.array(["mercado", "mercado", "outros", "outros"])
    y_pred = np.array(["mercado", "mercado", "outros", "outros"])
    # Scores: a classe mercado recebe scores altos para os positivos.
    scores = np.array(
        [
            [0.9, 0.1],
            [0.8, 0.2],
            [0.2, 0.8],
            [0.1, 0.9],
        ]
    )
    m = calcular_metricas_fold(y_true, y_pred, scores, classes)
    assert m["pr_auc_binario_projetado"] == 1.0


def test_pr_auc_none_quando_classe_positiva_ausente() -> None:
    classes = ["poder", "outros"]
    y_true = np.array(["poder", "outros"])
    y_pred = np.array(["poder", "outros"])
    scores = np.array([[0.9, 0.1], [0.1, 0.9]])
    m = calcular_metricas_fold(y_true, y_pred, scores, classes)
    assert m["pr_auc_binario_projetado"] is None


def test_composicao_fp_agrupa_por_y_original() -> None:
    classes = ["mercado", "outros"]
    y_true = np.array(["outros", "outros", "outros", "mercado"])
    y_pred = np.array(["mercado", "mercado", "outros", "mercado"])
    y_original = np.array(["poder", "asmais", "outros", "mercado"])
    scores = np.full((4, 2), 0.5)
    m = calcular_metricas_fold(y_true, y_pred, scores, classes, y_original=y_original)
    assert m["composicao_fp_mercado_por_y_original"] == {"poder": 1, "asmais": 1}


def test_bootstrap_ic_cobre_media() -> None:
    import math

    valores = [0.70, 0.72, 0.74, 0.68, 0.71]
    ic = bootstrap_ic(valores, n=500, seed=2026)
    assert math.isclose(ic["media"], sum(valores) / len(valores), rel_tol=1e-9)
    assert ic["ic_95_inf"] <= ic["media"] <= ic["ic_95_sup"]


def test_resumir_cross_fold_produz_ic_por_metrica() -> None:
    metricas = [
        {
            "f1_binario_projetado": 0.7 + i * 0.01,
            "pr_auc_binario_projetado": 0.8 + i * 0.005,
            "macro_f1": 0.6 + i * 0.01,
            "n_teste": 100,
        }
        for i in range(5)
    ]
    resumo = resumir_cross_fold(metricas)
    assert "f1_binario_projetado" in resumo
    assert "pr_auc_binario_projetado" in resumo
    assert "macro_f1" in resumo
    assert resumo["n_folds"] == 5
