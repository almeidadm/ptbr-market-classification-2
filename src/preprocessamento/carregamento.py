"""
Carregamento do corpus FolhaUOL para a pipeline experimental.

Difere de `src.eda.carregamento`:
- Filtra linhas com `date` ou `text` nulos (pré-requisito para split
  temporal, ADR 008; e para qualquer inferência textual).
- Renomeia `category` → `y_original` para uniformizar o vocabulário
  usado nos recortes (ADR 004) e nos artefatos (ADR 009).
- Preserva `link` como identificador único do artigo (chave de junção
  com diagnóstico de near-dup).

Não aplica dedup nem recortes — esses passos ficam em módulos próprios
para que cada transformação do corpus seja testável e rastreável.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.eda.carregamento import carregar_articles


COLUNAS_SAIDA = ("title", "text", "date", "y_original", "link", "id_raw")


def preparar_corpus(caminho_csv: Path) -> pd.DataFrame:
    """
    Carrega `articles.csv`, filtra nulos e renomeia a coluna-alvo.

    A coluna `id_raw` registra a posição de cada linha no corpus bruto
    pós-ordenação estável (antes do filtro de nulos). É essa numeração
    que o CSV `artifacts/eda/tabelas/07-near-duplicates-pares.csv` usa
    em `id_a`/`id_b`, então precisa ser preservada ao longo de toda a
    pipeline para que o diagnóstico de leakage funcione.
    """
    df = carregar_articles(caminho_csv)
    df["id_raw"] = range(len(df))
    df = df.dropna(subset=["date", "text"]).reset_index(drop=True)
    df = df.rename(columns={"category": "y_original"})
    return df[list(COLUNAS_SAIDA)].reset_index(drop=True)
