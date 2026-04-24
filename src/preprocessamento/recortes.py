"""
Recortes de classes do corpus (ADR 004).

Três recortes:
- Opção 7 (primário): top-6 concorrentes + `mercado` + `outros` (colapso
  da cauda), 8 classes. Preserva TODAS as linhas do corpus.
- Opção 4 (ablation): todas as categorias com ≥ 500 amostras, exceto
  `colunas` e `ilustrada`, 21 classes. Filtra linhas fora dessa lista.
- Opção 3 (ablation Garcia-style): 5 classes específicas. Filtra.

Em todos os recortes, `y_original` é preservada intocada e `y_colapsado`
carrega o rótulo efetivamente usado pelo modelo.
"""
from __future__ import annotations

import pandas as pd

from src.config import (
    CLASSES_OPCAO_3,
    CLASSES_OPCAO_7,
    THRESHOLD_OPCAO_4,
    THRESHOLD_OPCAO_7,
)


def aplicar_opcao_7(
    df: pd.DataFrame,
    threshold: int = THRESHOLD_OPCAO_7,
) -> pd.DataFrame:
    """
    Threshold ≥ `threshold` + relabel (ADR 004 §D.5 item 7).

    Duas operações em sequência:
    1. Remove linhas cuja `y_original` tenha contagem < `threshold`
       no corpus de entrada (categorias não-treináveis: `musica=1`,
       `bichos=1`, etc. citadas na ADR 004).
    2. Nas linhas restantes, colapsa toda `y_original` que não seja
       uma das 7 âncoras (`{poder, colunas, mercado, esporte, mundo,
       cotidiano, ilustrada}`) em `outros`.
    """
    contagens = df["y_original"].value_counts()
    classes_acima_do_threshold = set(contagens[contagens >= threshold].index)
    df_filtrado = df[df["y_original"].isin(classes_acima_do_threshold)].copy()

    ancoras = set(CLASSES_OPCAO_7) - {"outros"}
    df_filtrado["y_colapsado"] = df_filtrado["y_original"].where(
        df_filtrado["y_original"].isin(ancoras),
        other="outros",
    )
    return df_filtrado.reset_index(drop=True)


def aplicar_opcao_4(
    df: pd.DataFrame,
    threshold: int = THRESHOLD_OPCAO_4,
) -> pd.DataFrame:
    """
    Filtra categorias com ≥ `threshold` amostras, excluindo `colunas` e
    `ilustrada`. `y_colapsado` = `y_original` (sem relabel).

    O conjunto efetivo de classes é função do corpus de entrada — quando
    chamado sobre o corpus pós-dedup, produz as 21 classes canônicas da
    ADR 004 §D.5 (pendente de enumeração explícita em `enumeracao.py`).
    """
    contagens = df["y_original"].value_counts()
    candidatas = set(contagens[contagens >= threshold].index)
    classes_validas = candidatas - {"colunas", "ilustrada"}

    df_filtrado = df[df["y_original"].isin(classes_validas)].copy()
    df_filtrado["y_colapsado"] = df_filtrado["y_original"]
    return df_filtrado.reset_index(drop=True)


def aplicar_opcao_3(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filtra para as 5 classes Garcia-style declaradas em `config.CLASSES_OPCAO_3`.
    """
    classes = set(CLASSES_OPCAO_3)
    df_filtrado = df[df["y_original"].isin(classes)].copy()
    df_filtrado["y_colapsado"] = df_filtrado["y_original"]
    return df_filtrado.reset_index(drop=True)
