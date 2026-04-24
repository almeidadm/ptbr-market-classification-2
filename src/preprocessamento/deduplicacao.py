"""
Deduplicação por texto exato (ADR 007).

Regra: dentro de cada grupo de linhas com `text` idêntico, manter a
linha de `date` mais antiga; em caso de empate, manter a de `link`
lexicograficamente menor. As linhas removidas são descartadas do corpus
efetivo, mas cada linha sobrevivente recebe a flag `eh_duplicata_text`
indicando se fazia parte de um grupo com ≥ 2 instâncias — a flag serve
como coluna de auditoria em análises posteriores (não filtra nada).
"""
from __future__ import annotations

import pandas as pd


def deduplicar_por_texto_exato(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove duplicatas exatas de `text` e anexa `eh_duplicata_text`.

    A ordenação prévia `(text, date, link)` com `mergesort` é o que
    garante que `drop_duplicates(keep="first")` preserve a linha mais
    antiga com desempate lexicográfico por `link`.
    """
    contagens_por_texto = df.groupby("text", sort=False).size()
    textos_repetidos = set(contagens_por_texto[contagens_por_texto >= 2].index)

    df_ordenado = df.sort_values(
        by=["text", "date", "link"],
        kind="mergesort",
    )
    df_dedup = df_ordenado.drop_duplicates(subset="text", keep="first")
    df_dedup = df_dedup.copy()
    df_dedup["eh_duplicata_text"] = df_dedup["text"].isin(textos_repetidos)

    return df_dedup.sort_values(
        by=["date", "link"],
        kind="mergesort",
    ).reset_index(drop=True)
