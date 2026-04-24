"""
Métricas de avaliação para os experimentos (ADR 006 + ADR 009 §D.6).

Primárias:
- F1 binária projetada de `mercado`: `y_pred ∈ classes_multi` é colapsado
  para `{mercado, resto}` e comparado contra `y_true` sob o mesmo mapa.
- PR-AUC binária projetada: `score_mercado` extraído da coluna da classe
  `mercado` em `scores` (probs ou `decision_function`), comparado contra
  `y_true_bin`.

Secundárias:
- Macro-F1 multiclasse.
- Matriz de confusão (`classes × classes`).
- Composição dos falsos positivos de `mercado` por `y_original` (A1 da
  ADR 004) — a razão para preservar `y_original` em todos os recortes.

Agregação cross-fold: média simples + IC 95% por bootstrap (n=1000,
seed=2026) sobre o vetor de valores por fold.
"""
from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd
from sklearn.metrics import (
    auc,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
)

from src.config import BOOTSTRAP_N, CLASSE_POSITIVA, SEED


def projetar_binario(y: np.ndarray | pd.Series) -> np.ndarray:
    """Converte rótulos multiclasse em 1 (classe positiva) / 0 (resto)."""
    valores = np.asarray(y)
    return (valores == CLASSE_POSITIVA).astype(int)


def _score_classe_positiva(
    scores: np.ndarray, classes: list[str]
) -> np.ndarray | None:
    """Extrai a coluna correspondente a `mercado`; `None` se ausente."""
    if CLASSE_POSITIVA not in classes:
        return None
    indice = classes.index(CLASSE_POSITIVA)
    return scores[:, indice]


def calcular_metricas_fold(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    scores: np.ndarray,
    classes: list[str],
    y_original: np.ndarray | pd.Series | None = None,
) -> dict:
    """
    Computa o bloco de métricas de um único fold.

    `y_true` e `y_pred` estão no espaço multiclasse do recorte;
    `scores` tem forma `(n, |classes|)`. `y_original` é a coluna
    preservada da ADR 004 — quando presente, viabiliza a composição
    de falsos positivos por subcategoria (análise A1).
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    y_true_bin = projetar_binario(y_true)
    y_pred_bin = projetar_binario(y_pred)

    f1_bin = float(f1_score(y_true_bin, y_pred_bin, pos_label=1, zero_division=0))
    pr_auc_bin = _calcular_pr_auc(y_true_bin, scores, classes)
    macro_f1 = float(
        f1_score(y_true, y_pred, average="macro", labels=classes, zero_division=0)
    )
    matriz = confusion_matrix(y_true, y_pred, labels=classes).tolist()

    composicao_fp = _composicao_fp_por_y_original(
        y_true=y_true, y_pred=y_pred, y_original=y_original
    )

    return {
        "f1_binario_projetado": f1_bin,
        "pr_auc_binario_projetado": pr_auc_bin,
        "macro_f1": macro_f1,
        "matriz_confusao": matriz,
        "classes": list(classes),
        "composicao_fp_mercado_por_y_original": composicao_fp,
        "n_teste": int(len(y_true)),
        "n_mercado_teste": int(y_true_bin.sum()),
    }


def _calcular_pr_auc(
    y_true_bin: np.ndarray, scores: np.ndarray, classes: list[str]
) -> float | None:
    coluna = _score_classe_positiva(scores, classes)
    if coluna is None:
        return None
    if y_true_bin.sum() == 0 or y_true_bin.sum() == len(y_true_bin):
        return None
    precision, recall, _ = precision_recall_curve(y_true_bin, coluna)
    return float(auc(recall, precision))


def _composicao_fp_por_y_original(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_original: np.ndarray | pd.Series | None,
) -> dict[str, int] | None:
    if y_original is None:
        return None
    y_original = np.asarray(y_original)
    mascara_fp = (y_pred == CLASSE_POSITIVA) & (y_true != CLASSE_POSITIVA)
    if mascara_fp.sum() == 0:
        return {}
    contagem = Counter(y_original[mascara_fp].tolist())
    return {str(k): int(v) for k, v in sorted(contagem.items(), key=lambda kv: -kv[1])}


def bootstrap_ic(
    valores: list[float] | np.ndarray, n: int = BOOTSTRAP_N, seed: int = SEED
) -> dict:
    """Média + IC 95% por bootstrap. `valores` tipicamente é uma
    métrica por fold (5 pontos). O resampling com reposição sobre 5
    pontos é grosseiro, mas segue a prática da ADR 009 §D.6."""
    valores = np.asarray(valores, dtype=float)
    if len(valores) == 0:
        return {"media": None, "ic_95_inf": None, "ic_95_sup": None, "dp": None}

    rng = np.random.default_rng(seed)
    amostras = rng.choice(valores, size=(n, len(valores)), replace=True).mean(axis=1)
    return {
        "media": float(valores.mean()),
        "dp": float(valores.std(ddof=1)) if len(valores) > 1 else 0.0,
        "ic_95_inf": float(np.percentile(amostras, 2.5)),
        "ic_95_sup": float(np.percentile(amostras, 97.5)),
    }


def resumir_cross_fold(metricas_por_fold: list[dict]) -> dict:
    """Agrega métricas escalares com média + IC 95% via bootstrap."""
    if not metricas_por_fold:
        return {}

    def _extrair(chave: str) -> list[float]:
        return [m[chave] for m in metricas_por_fold if m.get(chave) is not None]

    return {
        "f1_binario_projetado": bootstrap_ic(_extrair("f1_binario_projetado")),
        "pr_auc_binario_projetado": bootstrap_ic(_extrair("pr_auc_binario_projetado")),
        "macro_f1": bootstrap_ic(_extrair("macro_f1")),
        "n_folds": len(metricas_por_fold),
        "seed_bootstrap": SEED,
        "n_bootstrap": BOOTSTRAP_N,
    }
