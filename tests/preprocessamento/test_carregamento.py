"""Testes para `src.preprocessamento.carregamento`."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.preprocessamento.carregamento import COLUNAS_SAIDA, preparar_corpus


def _escrever_csv(tmp_path: Path, linhas: list[dict]) -> Path:
    df = pd.DataFrame(linhas, columns=["title", "text", "date", "category", "subcategory", "link"])
    caminho = tmp_path / "articles.csv"
    df.to_csv(caminho, index=False)
    return caminho


def test_colunas_de_saida_sao_as_canonicas(tmp_path: Path) -> None:
    csv = _escrever_csv(
        tmp_path,
        [
            {"title": "t1", "text": "corpo 1", "date": "2016-01-01",
             "category": "Mercado", "subcategory": "", "link": "a"},
        ],
    )
    df = preparar_corpus(csv)
    assert list(df.columns) == list(COLUNAS_SAIDA)
    assert "y_original" in df.columns
    assert "category" not in df.columns


def test_remove_linhas_com_date_ou_text_nulos(tmp_path: Path) -> None:
    csv = _escrever_csv(
        tmp_path,
        [
            {"title": "t1", "text": "corpo 1", "date": "2016-01-01",
             "category": "mercado", "subcategory": "", "link": "a"},
            {"title": "t2", "text": None, "date": "2016-01-02",
             "category": "mercado", "subcategory": "", "link": "b"},
            {"title": "t3", "text": "corpo 3", "date": None,
             "category": "poder", "subcategory": "", "link": "c"},
            {"title": "t4", "text": "corpo 4", "date": "2016-01-05",
             "category": "Esporte", "subcategory": "", "link": "d"},
        ],
    )
    df = preparar_corpus(csv)
    assert len(df) == 2
    assert set(df["link"]) == {"a", "d"}


def test_ordenacao_estavel_por_date_link(tmp_path: Path) -> None:
    csv = _escrever_csv(
        tmp_path,
        [
            {"title": "t1", "text": "c1", "date": "2016-03-01",
             "category": "mercado", "subcategory": "", "link": "z"},
            {"title": "t2", "text": "c2", "date": "2016-01-15",
             "category": "poder", "subcategory": "", "link": "b"},
            {"title": "t3", "text": "c3", "date": "2016-01-15",
             "category": "poder", "subcategory": "", "link": "a"},
        ],
    )
    df = preparar_corpus(csv)
    assert df["link"].tolist() == ["a", "b", "z"]


def test_y_original_normalizada_strip_lower_sem_acentos(tmp_path: Path) -> None:
    csv = _escrever_csv(
        tmp_path,
        [
            {"title": "t", "text": "c", "date": "2016-01-01",
             "category": " São Paulo ", "subcategory": "", "link": "x"},
        ],
    )
    df = preparar_corpus(csv)
    assert df.loc[0, "y_original"] == "sao paulo"


def test_id_raw_preserva_posicao_no_corpus_bruto(tmp_path: Path) -> None:
    csv = _escrever_csv(
        tmp_path,
        [
            # Ordem estável pós-carregamento: (2016-01-01, b), (2016-01-15, a), (2016-03-01, z).
            # Índices pós-ordenação antes do dropna: b=0, a=1, z=2.
            {"title": "t1", "text": "c1", "date": "2016-03-01",
             "category": "mercado", "subcategory": "", "link": "z"},
            {"title": "t2", "text": None, "date": "2016-02-10",
             "category": "mercado", "subcategory": "", "link": "nulo"},
            {"title": "t3", "text": "c3", "date": "2016-01-15",
             "category": "poder", "subcategory": "", "link": "a"},
            {"title": "t4", "text": "c4", "date": "2016-01-01",
             "category": "esporte", "subcategory": "", "link": "b"},
        ],
    )
    df = preparar_corpus(csv)
    # Linha com text=None é removida, mas id_raw dos sobreviventes
    # continua sendo posição no corpus bruto ordenado.
    por_link = dict(zip(df["link"], df["id_raw"]))
    assert por_link == {"b": 0, "a": 1, "z": 3}
