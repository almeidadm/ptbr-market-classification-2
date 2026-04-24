"""
Rolling-origin expanding window (ADR 008, protocolo primário).

Produz 5 folds temporais a partir de `df["date"]`:

- Janela inicial de treino: 18 meses a partir do início do corpus.
- Cada fold de teste consome os 3 meses imediatamente seguintes.
- Janela de treino cresce (expanding), acumulando todos os meses
  anteriores ao início do teste.
- Inner val: últimos 3 meses do treino de cada fold (split temporal
  para a busca de hiperparâmetros da ADR 009 §D.2).

A função é determinística: depende apenas dos valores de `date` e não
usa aleatoriedade.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import (
    INNER_VAL_MESES,
    ROLLING_JANELA_TESTE_MESES,
    ROLLING_JANELA_TREINO_INICIAL_MESES,
    ROLLING_N_FOLDS,
)
from src.splitting.fold import Fold


def gerar_folds_rolling(
    df: pd.DataFrame,
    janela_treino_inicial_meses: int = ROLLING_JANELA_TREINO_INICIAL_MESES,
    janela_teste_meses: int = ROLLING_JANELA_TESTE_MESES,
    n_folds: int = ROLLING_N_FOLDS,
    inner_val_meses: int = INNER_VAL_MESES,
) -> list[Fold]:
    """
    Gera a lista de folds conforme o protocolo primário.

    O alinhamento temporal é feito a partir do início do mês da data
    mínima em `df["date"]`; isso evita que pequenas variações no dia
    da primeira notícia desloquem toda a grade de folds.
    """
    if "date" not in df.columns:
        raise ValueError("DataFrame precisa ter a coluna `date`.")
    if df["date"].isna().any():
        raise ValueError("`date` não pode ter valores nulos ao gerar folds.")

    datas = pd.to_datetime(df["date"])
    inicio_corpus = datas.min().to_period("M").to_timestamp()

    folds: list[Fold] = []
    for indice_fold in range(n_folds):
        inicio_teste = inicio_corpus + pd.DateOffset(
            months=janela_treino_inicial_meses + indice_fold * janela_teste_meses
        )
        fim_teste = inicio_teste + pd.DateOffset(months=janela_teste_meses)

        mascara_treino = datas < inicio_teste
        mascara_teste = (datas >= inicio_teste) & (datas < fim_teste)

        treino_idx = np.flatnonzero(mascara_treino.values)
        teste_idx = np.flatnonzero(mascara_teste.values)

        if len(teste_idx) == 0:
            raise ValueError(
                f"Fold {indice_fold} com teste vazio "
                f"({inicio_teste.date()}–{fim_teste.date()}): "
                f"corpus não cobre essa janela temporal."
            )

        inicio_inner_val = inicio_teste - pd.DateOffset(months=inner_val_meses)
        datas_treino = datas.iloc[treino_idx]
        mascara_inner_val = datas_treino >= inicio_inner_val

        posicoes_treino = np.arange(len(treino_idx))
        inner_val_idx = treino_idx[mascara_inner_val.values]
        inner_train_idx = treino_idx[~mascara_inner_val.values]

        if len(inner_val_idx) == 0 or len(inner_train_idx) == 0:
            raise ValueError(
                f"Fold {indice_fold}: inner split temporal produziu "
                f"partição vazia (inner_val_meses={inner_val_meses})."
            )

        folds.append(
            Fold(
                indice=indice_fold,
                treino_idx=treino_idx,
                teste_idx=teste_idx,
                inner_train_idx=inner_train_idx,
                inner_val_idx=inner_val_idx,
                inicio_teste=inicio_teste,
                fim_teste=fim_teste,
            )
        )

    return folds
