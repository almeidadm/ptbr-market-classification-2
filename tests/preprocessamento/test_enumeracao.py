"""Testes para `src.preprocessamento.enumeracao`."""
from __future__ import annotations

import pandas as pd

from src.preprocessamento.enumeracao import (
    enumerar_classes_opcao_4,
    tabular_contagens_opcao_4,
)


def _corpus() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "y_original": (
                ["poder"] * 1000
                + ["mercado"] * 800
                + ["esporte"] * 600
                + ["colunas"] * 900
                + ["ilustrada"] * 700
                + ["tec"] * 400
                + ["bbc"] * 550
            ),
        }
    )


def test_enumerar_exclui_colunas_ilustrada_e_abaixo_do_threshold() -> None:
    classes = enumerar_classes_opcao_4(_corpus(), threshold=500)
    assert classes == sorted(["poder", "mercado", "esporte", "bbc"])


def test_enumerar_determinismo() -> None:
    df = _corpus()
    c1 = enumerar_classes_opcao_4(df, threshold=500)
    c2 = enumerar_classes_opcao_4(df, threshold=500)
    assert c1 == c2


def test_tabular_contagens_bate_soma_e_proporcao() -> None:
    tabela = tabular_contagens_opcao_4(_corpus(), threshold=500)
    assert list(tabela.columns) == ["classe", "n", "frac"]
    assert tabela["n"].sum() == 1000 + 800 + 600 + 550
    assert abs(tabela["frac"].sum() - 1.0) < 1e-9


def test_enumerar_vazio_quando_todas_abaixo_do_threshold() -> None:
    df = pd.DataFrame({"y_original": ["x"] * 10 + ["y"] * 5})
    assert enumerar_classes_opcao_4(df, threshold=500) == []
