"""
Stratified k-fold cross-validation (ADR 008, ablation).

Ablation intencionalmente cega ao tempo: embaralha o corpus inteiro e
particiona respeitando a distribuição de `y_colapsado`. O contraste
com o rolling-origin quantifica o otimismo do protocolo clássico
frente à deriva documentada na ADR 003.

Inner val: 20% estratificado do treino de cada fold, seed fixa. A
ablation **não** pretende ser temporalmente honesta; seu objetivo é
produzir o pior caso de vazamento para comparação.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

from src.config import KFOLD_K, SEED
from src.splitting.fold import Fold


INNER_VAL_FRAC = 0.2


def gerar_folds_kfold(
    df: pd.DataFrame,
    coluna_estratificacao: str = "y_colapsado",
    k: int = KFOLD_K,
    seed: int = SEED,
    inner_val_frac: float = INNER_VAL_FRAC,
) -> list[Fold]:
    """
    Gera a lista de folds estratificados por `coluna_estratificacao`.
    """
    if coluna_estratificacao not in df.columns:
        raise ValueError(
            f"Coluna `{coluna_estratificacao}` ausente; não há como estratificar."
        )

    y = df[coluna_estratificacao].to_numpy()
    estratificador = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)

    folds: list[Fold] = []
    for indice_fold, (treino_idx, teste_idx) in enumerate(
        estratificador.split(np.zeros(len(df)), y)
    ):
        y_treino = y[treino_idx]
        inner_train_idx, inner_val_idx = train_test_split(
            treino_idx,
            test_size=inner_val_frac,
            random_state=seed,
            stratify=y_treino,
        )
        folds.append(
            Fold(
                indice=indice_fold,
                treino_idx=np.asarray(treino_idx),
                teste_idx=np.asarray(teste_idx),
                inner_train_idx=np.asarray(inner_train_idx),
                inner_val_idx=np.asarray(inner_val_idx),
                inicio_teste=None,
                fim_teste=None,
            )
        )
    return folds
