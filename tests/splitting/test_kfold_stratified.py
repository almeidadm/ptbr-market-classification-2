"""Testes para `src.splitting.kfold_stratified` (ADR 008 ablation)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.splitting.kfold_stratified import gerar_folds_kfold


def _corpus_com_desbalanceamento(n: int = 500) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    classes = rng.choice(
        ["mercado", "poder", "esporte", "outros"],
        size=n,
        p=[0.125, 0.2, 0.15, 0.525],
    )
    return pd.DataFrame(
        {
            "y_colapsado": classes,
            "link": [f"l{i}" for i in range(n)],
        }
    )


def test_gera_cinco_folds_com_particao_exaustiva() -> None:
    df = _corpus_com_desbalanceamento()
    folds = gerar_folds_kfold(df)
    assert len(folds) == 5

    uniao_teste = np.concatenate([f.teste_idx for f in folds])
    assert len(uniao_teste) == len(df)
    assert len(np.unique(uniao_teste)) == len(df)


def test_treino_e_teste_disjuntos() -> None:
    df = _corpus_com_desbalanceamento()
    folds = gerar_folds_kfold(df)
    for fold in folds:
        assert len(np.intersect1d(fold.treino_idx, fold.teste_idx)) == 0


def test_inner_train_e_inner_val_disjuntos_e_cobrem_treino() -> None:
    df = _corpus_com_desbalanceamento()
    folds = gerar_folds_kfold(df)
    for fold in folds:
        assert len(np.intersect1d(fold.inner_train_idx, fold.inner_val_idx)) == 0
        juntos = np.sort(np.concatenate([fold.inner_train_idx, fold.inner_val_idx]))
        np.testing.assert_array_equal(juntos, np.sort(fold.treino_idx))


def test_determinismo_sob_mesma_seed() -> None:
    df = _corpus_com_desbalanceamento()
    f1 = gerar_folds_kfold(df, seed=2026)
    f2 = gerar_folds_kfold(df, seed=2026)
    for a, b in zip(f1, f2):
        np.testing.assert_array_equal(a.treino_idx, b.treino_idx)
        np.testing.assert_array_equal(a.teste_idx, b.teste_idx)


def test_seeds_diferentes_produzem_folds_diferentes() -> None:
    df = _corpus_com_desbalanceamento()
    f1 = gerar_folds_kfold(df, seed=1)
    f2 = gerar_folds_kfold(df, seed=2)
    assert not np.array_equal(f1[0].teste_idx, f2[0].teste_idx)


def test_prevalencia_aproximadamente_preservada_em_cada_fold() -> None:
    df = _corpus_com_desbalanceamento(n=1000)
    prevalencia_global = (df["y_colapsado"] == "mercado").mean()
    folds = gerar_folds_kfold(df)
    for fold in folds:
        prev_teste = (df.iloc[fold.teste_idx]["y_colapsado"] == "mercado").mean()
        assert abs(prev_teste - prevalencia_global) < 0.05
