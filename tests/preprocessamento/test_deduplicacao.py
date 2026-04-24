"""Testes para `src.preprocessamento.deduplicacao` (ADR 007)."""
from __future__ import annotations

import pandas as pd
import pytest

from src.preprocessamento.deduplicacao import deduplicar_por_texto_exato


def _df(linhas: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(linhas)
    df["date"] = pd.to_datetime(df["date"])
    return df


def test_grupo_de_tres_duplicatas_mantem_mais_antiga() -> None:
    df = _df(
        [
            {"text": "A", "date": "2016-03-01", "link": "a3", "y_original": "mercado"},
            {"text": "A", "date": "2016-01-15", "link": "a1", "y_original": "mercado"},
            {"text": "A", "date": "2016-02-10", "link": "a2", "y_original": "mercado"},
            {"text": "B", "date": "2016-04-01", "link": "b1", "y_original": "poder"},
        ]
    )
    dedup = deduplicar_por_texto_exato(df)
    assert len(dedup) == 2
    assert set(dedup["link"]) == {"a1", "b1"}


def test_desempate_por_link_lexicografico_em_date_igual() -> None:
    df = _df(
        [
            {"text": "X", "date": "2016-01-10", "link": "z"},
            {"text": "X", "date": "2016-01-10", "link": "a"},
            {"text": "X", "date": "2016-01-10", "link": "m"},
        ]
    )
    dedup = deduplicar_por_texto_exato(df)
    assert len(dedup) == 1
    assert dedup.loc[0, "link"] == "a"


def test_flag_eh_duplicata_text_marca_sobreviventes_de_grupos_repetidos() -> None:
    df = _df(
        [
            {"text": "A", "date": "2016-01-01", "link": "a"},
            {"text": "A", "date": "2016-01-02", "link": "b"},
            {"text": "B", "date": "2016-01-03", "link": "c"},
            {"text": "C", "date": "2016-01-04", "link": "d"},
            {"text": "C", "date": "2016-01-05", "link": "e"},
            {"text": "D", "date": "2016-01-06", "link": "f"},
        ]
    )
    dedup = deduplicar_por_texto_exato(df)
    # Sobreviventes: a (grupo 2), c (singleton), d (grupo 2), f (singleton)
    flags = dict(zip(dedup["link"], dedup["eh_duplicata_text"]))
    assert flags == {"a": True, "c": False, "d": True, "f": False}


def test_ordem_final_por_date_link_preservada() -> None:
    df = _df(
        [
            {"text": "A", "date": "2016-03-01", "link": "a"},
            {"text": "B", "date": "2016-01-15", "link": "b"},
            {"text": "C", "date": "2016-02-10", "link": "c"},
        ]
    )
    dedup = deduplicar_por_texto_exato(df)
    assert dedup["link"].tolist() == ["b", "c", "a"]


def test_determinismo_sob_duas_execucoes() -> None:
    df = _df(
        [
            {"text": "A", "date": "2016-01-10", "link": "x"},
            {"text": "A", "date": "2016-01-10", "link": "y"},
            {"text": "B", "date": "2016-02-10", "link": "z"},
        ]
    )
    d1 = deduplicar_por_texto_exato(df)
    d2 = deduplicar_por_texto_exato(df)
    pd.testing.assert_frame_equal(d1, d2)


def test_nao_reduz_corpus_sem_duplicatas() -> None:
    df = _df(
        [
            {"text": f"texto {i}", "date": f"2016-01-{i:02d}", "link": f"l{i}"}
            for i in range(1, 6)
        ]
    )
    dedup = deduplicar_por_texto_exato(df)
    assert len(dedup) == 5
    assert not dedup["eh_duplicata_text"].any()
