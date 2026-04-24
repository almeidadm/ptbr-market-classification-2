"""
Enumeração das classes do recorte Opção 4 a partir do corpus pós-dedup.

Resolve a pendência da ADR 004: o conjunto exato das ~21 classes do
recorte Opção 4 depende da tabulação empírica do corpus após a
deduplicação da ADR 007. Este módulo materializa essa enumeração de
forma pura e determinística.
"""
from __future__ import annotations

import pandas as pd

from src.config import THRESHOLD_OPCAO_4


def enumerar_classes_opcao_4(
    df: pd.DataFrame,
    threshold: int = THRESHOLD_OPCAO_4,
) -> list[str]:
    """
    Lista ordenada alfabeticamente das classes do recorte Opção 4.

    Inclui toda categoria com contagem ≥ `threshold` em `y_original`,
    excluindo `colunas` e `ilustrada` (ADR 004 §D.5).
    """
    contagens = df["y_original"].value_counts()
    candidatas = set(contagens[contagens >= threshold].index)
    classes = sorted(candidatas - {"colunas", "ilustrada"})
    return classes


def tabular_contagens_opcao_4(
    df: pd.DataFrame,
    threshold: int = THRESHOLD_OPCAO_4,
) -> pd.DataFrame:
    """
    Tabela com `classe, n, frac` para auditoria e para preenchimento das
    descrições curtas no prompt v1 definitivo do recorte Opção 4.
    """
    classes = enumerar_classes_opcao_4(df, threshold=threshold)
    sub = df[df["y_original"].isin(classes)]
    contagens = sub["y_original"].value_counts().reindex(classes).fillna(0).astype(int)
    total = int(contagens.sum())
    tabela = pd.DataFrame(
        {
            "classe": contagens.index,
            "n": contagens.values,
            "frac": (contagens.values / total) if total > 0 else 0.0,
        }
    )
    return tabela.reset_index(drop=True)
