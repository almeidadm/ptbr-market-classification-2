"""Testes leves para `src.eda.carregamento`."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.eda.carregamento import carregar_articles, eh_mercado, normalizar_categoria


def test_normalizar_categoria_strip_lower_e_acentos() -> None:
    assert normalizar_categoria(" Mercado ") == "mercado"
    assert normalizar_categoria("MERCADO") == "mercado"
    # "mercádo" tem acento artificial — normalização deve removê-lo.
    assert normalizar_categoria("mercádo") == "mercado"


def test_eh_mercado_em_variantes() -> None:
    serie = pd.Series([" Mercado ", "MERCADO", "mercádo", "esporte", None])
    mascara = eh_mercado(serie)
    assert mascara.tolist() == [True, True, True, False, False]


def test_carregar_articles_preserva_contagem_e_tipa_date(tmp_path: Path) -> None:
    csv_sintetico = tmp_path / "articles.csv"
    dados = pd.DataFrame(
        {
            "title": ["t1", "t2", "t3", "t4"],
            "text": ["x1", "x2", "x3", "x4"],
            "date": ["2016-01-01", "2016-06-15", "2017-03-01", "2015-12-31"],
            "category": ["Mercado", "esporte", "MERCADO", "mercádo"],
            "subcategory": ["a", "b", "c", "d"],
            "link": ["l4", "l3", "l2", "l1"],
        }
    )
    dados.to_csv(csv_sintetico, index=False)

    df = carregar_articles(csv_sintetico)

    assert len(df) == 4
    assert df["date"].dtype.kind == "M"
    # Ordenação cronológica estável por (date, link).
    assert df["date"].is_monotonic_increasing
    assert df["category"].tolist() == ["mercado", "mercado", "esporte", "mercado"]
    # Todas as variantes de "mercado" foram capturadas pela normalização.
    mercado = df["category"].eq("mercado").sum()
    assert mercado == 3


def test_carregar_articles_falta_coluna(tmp_path: Path) -> None:
    csv_sintetico = tmp_path / "articles.csv"
    pd.DataFrame({"title": ["t"]}).to_csv(csv_sintetico, index=False)
    with pytest.raises(ValueError, match="Colunas esperadas ausentes"):
        carregar_articles(csv_sintetico)
