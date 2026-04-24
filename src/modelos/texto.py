"""
Preparação do texto de entrada (Q1 do ciclo 2026-04-24).

Convenção uniforme para todas as famílias (LogReg, SVM, BERTimbau,
Llama): concatenar `title` e `text` com um único espaço, tratando
nulos como string vazia. Sem separador explícito (`\\n`, `[SEP]`, etc.)
para manter o input idêntico entre modelos — diferenças entre famílias
ficam isoladas ao tokenizador.
"""
from __future__ import annotations

import pandas as pd


def juntar_titulo_texto(title: object, text: object) -> str:
    """Concatena título e corpo com espaço único, descartando nulos."""
    partes: list[str] = []
    if not pd.isna(title):
        t = str(title).strip()
        if t:
            partes.append(t)
    if not pd.isna(text):
        c = str(text).strip()
        if c:
            partes.append(c)
    return " ".join(partes)


def preparar_entrada(df: pd.DataFrame) -> pd.Series:
    """Aplica `juntar_titulo_texto` linha a linha, retornando uma Series."""
    return df.apply(
        lambda linha: juntar_titulo_texto(linha.get("title"), linha.get("text")),
        axis=1,
    )
