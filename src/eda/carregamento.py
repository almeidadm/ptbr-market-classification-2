"""
Carregamento e normalização do CSV bruto FolhaUOL para uso na EDA.

Normalização do alvo (decisão 002): a classe positiva é
`normalizar(category) == "mercado"`, onde `normalizar` aplica strip, lower
e remoção de acentos via `unicodedata.normalize('NFKD', ...)`. A coluna
`subcategory` é ignorada. `date` é tipada como `datetime64[ns]`.

Ordenação estável por `(date, link)` com `kind='mergesort'`: ordem
determinística é pré-requisito para reprodutibilidade dos agregados
temporais e dos splits futuros.
"""
from __future__ import annotations

import unicodedata
from pathlib import Path

import pandas as pd

COLUNAS_ESPERADAS = ["title", "text", "date", "category", "subcategory", "link"]


def _remover_acentos(texto: str) -> str:
    decomposto = unicodedata.normalize("NFKD", texto)
    return "".join(ch for ch in decomposto if not unicodedata.combining(ch))


def normalizar_categoria(valor: object) -> object:
    """Aplica strip -> lower -> remoção de acentos. Preserva nulos."""
    if pd.isna(valor):
        return valor
    return _remover_acentos(str(valor).strip().lower())


def eh_mercado(series: pd.Series) -> pd.Series:
    """Máscara booleana para a classe positiva sobre uma coluna de categorias.

    Aplica a mesma normalização do carregamento, de modo que a função
    funciona tanto sobre a coluna bruta quanto sobre uma coluna já
    normalizada. Nulos resultam em `False`.
    """
    normalizada = series.map(normalizar_categoria)
    return normalizada.eq("mercado").fillna(False)


def carregar_articles(caminho_csv: Path) -> pd.DataFrame:
    """Lê `articles.csv`, tipifica `date`, normaliza `category` e ordena.

    Não filtra linhas com `category` ou `date` nulos — a EDA (§1) tabula
    esses casos. A ordenação por `(date, link)` usa `mergesort` para ser
    estável, preservando a ordem de linhas com mesma data e link nulo.
    """
    caminho_csv = Path(caminho_csv)
    df = pd.read_csv(caminho_csv)

    faltando = [coluna for coluna in COLUNAS_ESPERADAS if coluna not in df.columns]
    if faltando:
        raise ValueError(
            f"Colunas esperadas ausentes no CSV {caminho_csv}: {faltando}"
        )

    df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d", errors="coerce")
    df["category"] = df["category"].map(normalizar_categoria)

    return df.sort_values(
        by=["date", "link"],
        kind="mergesort",
        na_position="last",
    ).reset_index(drop=True)
