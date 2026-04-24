"""Testes leves para `src.eda.contagens`."""
from __future__ import annotations

import pandas as pd
import pytest

from src.eda.contagens import (
    intervalo_wilson,
    prevalencia_por_dia_semana,
    prevalencia_por_mes_ano,
    prevalencia_por_periodo,
    serie_total_e_classe,
)


def test_prevalencia_por_periodo_mensal() -> None:
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2016-01-05",
                    "2016-01-20",
                    "2016-01-25",
                    "2016-02-10",
                    "2016-02-11",
                ]
            ),
            "is_mercado": [True, False, True, False, False],
        }
    )
    resultado = prevalencia_por_periodo(df, alvo="is_mercado", granularidade="M")

    assert list(resultado.columns) == ["total", "positivos", "prevalencia"]
    # Janeiro: 3 totais, 2 positivos -> 2/3.
    jan = resultado.iloc[0]
    assert jan["total"] == 3
    assert jan["positivos"] == 2
    assert jan["prevalencia"] == pytest.approx(2 / 3)
    # Fevereiro: 2 totais, 0 positivos -> 0.
    fev = resultado.iloc[1]
    assert fev["total"] == 2
    assert fev["positivos"] == 0
    assert fev["prevalencia"] == pytest.approx(0.0)


def test_prevalencia_por_periodo_granularidade_invalida() -> None:
    df = pd.DataFrame({"date": pd.to_datetime(["2016-01-01"]), "alvo": [True]})
    with pytest.raises(ValueError):
        prevalencia_por_periodo(df, alvo="alvo", granularidade="Y")  # type: ignore[arg-type]


def test_intervalo_wilson_contem_proporcao() -> None:
    p, baixo, alto = intervalo_wilson(positivos=126, total=1000)
    assert baixo < p < alto
    assert 0 < baixo < alto < 1
    # Faixa razoável para ~12,6% com n=1000.
    assert baixo > 0.10
    assert alto < 0.15


def test_intervalo_wilson_total_zero() -> None:
    p, baixo, alto = intervalo_wilson(positivos=0, total=0)
    assert pd.isna(p) and pd.isna(baixo) and pd.isna(alto)


@pytest.fixture
def df_multi_granularidade() -> pd.DataFrame:
    # 7 linhas distribuídas em 2 semanas / 2 meses para exercitar D, W e M.
    return pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2016-01-04",
                    "2016-01-04",
                    "2016-01-05",
                    "2016-01-11",
                    "2016-02-01",
                    "2016-02-01",
                    "2016-02-02",
                ]
            ),
            "is_mercado": [True, False, True, False, True, True, False],
        }
    )


def test_serie_total_e_classe_diaria(df_multi_granularidade: pd.DataFrame) -> None:
    resultado = serie_total_e_classe(df_multi_granularidade, "is_mercado", "D")
    assert list(resultado.columns) == ["total", "positivos", "prevalencia"]
    assert resultado.loc["2016-01-04", "total"] == 2
    assert resultado.loc["2016-01-04", "positivos"] == 1
    assert resultado.loc["2016-02-01", "positivos"] == 2


def test_serie_total_e_classe_semanal(df_multi_granularidade: pd.DataFrame) -> None:
    resultado = serie_total_e_classe(df_multi_granularidade, "is_mercado", "W")
    # `Grouper` preenche semanas sem observações com 0 — preservamos esse
    # comportamento para que o eixo temporal seja calendário completo.
    assert int(resultado["total"].sum()) == len(df_multi_granularidade)
    assert int(resultado["positivos"].sum()) == 4
    # Semanas com total == 0 geram `prevalencia` nula (divisão 0/0).
    assert resultado.loc[resultado["total"] == 0, "prevalencia"].isna().all()


def test_serie_total_e_classe_mensal(df_multi_granularidade: pd.DataFrame) -> None:
    resultado = serie_total_e_classe(df_multi_granularidade, "is_mercado", "M")
    assert list(resultado.index.month) == [1, 2]
    # Janeiro: 4 linhas, 2 positivas; Fevereiro: 3 linhas, 2 positivas.
    assert resultado["total"].tolist() == [4, 3]
    assert resultado["positivos"].tolist() == [2, 2]
    assert resultado["prevalencia"].tolist() == pytest.approx([0.5, 2 / 3])


def test_prevalencia_por_dia_semana_ordena_seg_dom() -> None:
    # 2016-01-04 = segunda, 2016-01-10 = domingo. Dois domingos com prevalência 100%,
    # uma segunda com prevalência 0% — só para checar ordenação e cálculo.
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2016-01-04",  # seg
                    "2016-01-10",  # dom
                    "2016-01-10",  # dom
                    "2016-01-17",  # dom
                ]
            ),
            "is_mercado": [False, True, True, True],
        }
    )
    resultado = prevalencia_por_dia_semana(df, "is_mercado")

    assert list(resultado.columns) == [
        "dia_semana",
        "nome",
        "total",
        "positivos",
        "prevalencia",
        "ic_inf",
        "ic_sup",
    ]
    # Apenas seg (0) e dom (6) aparecem; ordenação seg→dom está presente
    # implicitamente porque seg tem código menor.
    assert resultado["dia_semana"].tolist() == [0, 6]
    assert resultado["nome"].tolist() == ["seg", "dom"]
    assert resultado.loc[resultado["dia_semana"] == 0, "prevalencia"].iat[0] == 0.0
    assert resultado.loc[resultado["dia_semana"] == 6, "prevalencia"].iat[0] == 1.0
    # IC 95% Wilson contém o ponto observado.
    linha_dom = resultado.loc[resultado["dia_semana"] == 6].iloc[0]
    assert linha_dom["ic_inf"] < linha_dom["prevalencia"] <= linha_dom["ic_sup"]


def test_prevalencia_por_mes_ano_formato_longo() -> None:
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2016-01-05",
                    "2016-01-20",
                    "2016-02-10",
                    "2017-01-15",
                    "2017-01-16",
                ]
            ),
            "is_mercado": [True, False, True, True, True],
        }
    )
    resultado = prevalencia_por_mes_ano(df, "is_mercado")

    assert list(resultado.columns) == ["ano", "mes", "total", "positivos", "prevalencia"]
    # 3 células: (2016,1), (2016,2), (2017,1).
    assert len(resultado) == 3
    linha_2016_01 = resultado.loc[(resultado["ano"] == 2016) & (resultado["mes"] == 1)].iloc[0]
    assert linha_2016_01["total"] == 2
    assert linha_2016_01["positivos"] == 1
    assert linha_2016_01["prevalencia"] == pytest.approx(0.5)
    linha_2017_01 = resultado.loc[(resultado["ano"] == 2017) & (resultado["mes"] == 1)].iloc[0]
    assert linha_2017_01["prevalencia"] == pytest.approx(1.0)
