"""
Utilidades compartilhadas pelos scripts de execução de drift
(`scripts/drift/run_b1.py`, `run_b2.py`, futuros B3).

Mantém o carregamento de embeddings + corpus, a filtragem por escopo e
o empilhamento de embeddings por janela num único módulo importável da
biblioteca (`src/drift/`) — assim os scripts não precisam importar uns
dos outros via `from scripts.drift.run_b1 import ...`, evitando
dependência em namespace packages implícitos.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
    CLASSE_POSITIVA,
    DIR_ARTEFATOS_DRIFT,
    DIR_DADOS_PROCESSADO,
    ESCOPOS_DRIFT,
    GRANULARIDADES_DRIFT,
)
from src.drift.janelas import (
    Janela,
    gerar_janelas_bisemanais,
    gerar_janelas_mensais,
)


PADRAO_EMBEDDINGS: Path = (
    DIR_ARTEFATOS_DRIFT / "embeddings" / "bertimbau_base_cls" / "embeddings.parquet"
)
PADRAO_CORPUS: Path = DIR_DADOS_PROCESSADO / "corpus_opcao7.parquet"


def carregar_e_alinhar(
    caminho_embeddings: Path, caminho_corpus: Path
) -> pd.DataFrame:
    """
    Lê embeddings + corpus e devolve um único DataFrame ordenado por
    `date`, com colunas `link`, `date`, `y_original`, `embedding`.
    """
    if not caminho_embeddings.exists():
        raise FileNotFoundError(
            f"Embeddings não encontrados em {caminho_embeddings}. "
            "Rode scripts/drift/compute_embeddings.py antes."
        )
    if not caminho_corpus.exists():
        raise FileNotFoundError(
            f"Corpus não encontrado em {caminho_corpus}. "
            "Rode scripts/preprocessar.py antes."
        )

    emb = pd.read_parquet(caminho_embeddings, engine="pyarrow")
    corp = pd.read_parquet(
        caminho_corpus,
        engine="pyarrow",
        columns=["link", "y_original"],
    )
    df = emb.merge(corp, on="link", how="inner", validate="one_to_one")
    if len(df) != len(emb):
        raise ValueError(
            f"Join inconsistente: {len(emb)} embeddings vs {len(df)} após merge. "
            "Verifique se embeddings e corpus vêm da mesma rodada de dedup."
        )
    df = df.sort_values("date", kind="mergesort").reset_index(drop=True)
    return df


def filtrar_por_escopo(df: pd.DataFrame, escopo: str) -> pd.DataFrame:
    """`global` mantém tudo; `mercado` e `nao_mercado` filtram por `y_original`."""
    if escopo == "global":
        return df
    if escopo == "mercado":
        return df[df["y_original"] == CLASSE_POSITIVA].reset_index(drop=True)
    if escopo == "nao_mercado":
        return df[df["y_original"] != CLASSE_POSITIVA].reset_index(drop=True)
    raise ValueError(f"Escopo desconhecido: {escopo!r}. Esperado: {ESCOPOS_DRIFT}.")


def gerar_janelas(df: pd.DataFrame, granularidade: str) -> list[Janela]:
    if granularidade == "mensal":
        return gerar_janelas_mensais(df)
    if granularidade == "bisemanal":
        return gerar_janelas_bisemanais(df)
    raise ValueError(
        f"Granularidade desconhecida: {granularidade!r}. "
        f"Esperado: {GRANULARIDADES_DRIFT}."
    )


def empilhar_embeddings(df: pd.DataFrame, indices: np.ndarray) -> np.ndarray:
    """Concatena os embeddings dos índices num array (n, hidden_dim) float32."""
    return np.stack(df.iloc[indices]["embedding"].values).astype(np.float32)
