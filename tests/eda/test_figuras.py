"""Smoke tests para as figuras de §2.

Não comparamos pixels — validamos apenas que a função retorna uma
`matplotlib.figure.Figure` e que o título do eixo principal contém a
palavra-chave esperada. Suficiente para detectar regressões grosseiras
sem acoplar a testes a detalhes visuais.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.figure as mfigure
import pandas as pd
import pytest

from src.eda.figuras import (
    figura_contagem_diaria_com_anomalias,
    figura_heatmap_prevalencia_mes_ano,
    figura_prevalencia_dia_semana,
    figura_serie_diaria_prevalencia,
    figura_serie_mensal_contagens,
    figura_serie_mensal_prevalencia,
    figura_serie_semanal_prevalencia,
)


@pytest.fixture
def serie_sintetica_mensal() -> pd.DataFrame:
    datas = pd.date_range("2016-01-31", periods=6, freq="ME")
    return pd.DataFrame(
        {
            "total": [100, 120, 110, 130, 125, 140],
            "positivos": [12, 15, 14, 17, 16, 20],
            "prevalencia": [0.12, 0.125, 0.127, 0.131, 0.128, 0.143],
        },
        index=datas,
    )


def test_figura_serie_mensal_prevalencia(serie_sintetica_mensal: pd.DataFrame) -> None:
    figura = figura_serie_mensal_prevalencia(serie_sintetica_mensal)
    assert isinstance(figura, mfigure.Figure)
    (eixo,) = figura.axes
    assert "Prevalência" in eixo.get_title()


def test_figura_serie_mensal_contagens(serie_sintetica_mensal: pd.DataFrame) -> None:
    figura = figura_serie_mensal_contagens(serie_sintetica_mensal)
    assert isinstance(figura, mfigure.Figure)
    # Dois subplots empilhados: total em cima, positivos embaixo.
    assert len(figura.axes) == 2
    titulos = [eixo.get_title() for eixo in figura.axes]
    assert any("Total" in t for t in titulos)
    assert any("mercado" in t for t in titulos)


def test_figura_serie_semanal_prevalencia(serie_sintetica_mensal: pd.DataFrame) -> None:
    # Reutilizamos a série sintética mensal — a figura não diferencia freq,
    # só consome o índice temporal.
    figura = figura_serie_semanal_prevalencia(serie_sintetica_mensal)
    assert isinstance(figura, mfigure.Figure)
    (eixo,) = figura.axes
    assert "semanal" in eixo.get_title().lower()


def test_figura_serie_diaria_prevalencia(serie_sintetica_mensal: pd.DataFrame) -> None:
    figura = figura_serie_diaria_prevalencia(serie_sintetica_mensal)
    assert isinstance(figura, mfigure.Figure)
    (eixo,) = figura.axes
    assert "diária" in eixo.get_title().lower()


def test_figura_aceita_serie_sem_indice_temporal(
    serie_sintetica_mensal: pd.DataFrame,
) -> None:
    """Contrato do ciclo: tabelas CSV têm coluna `data`, não índice."""
    serie_csv = serie_sintetica_mensal.reset_index().rename(columns={"index": "data"})
    figura = figura_serie_mensal_prevalencia(serie_csv)
    assert isinstance(figura, mfigure.Figure)


@pytest.fixture
def df_dia_semana_sintetico() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "dia_semana": [0, 1, 2, 3, 4, 5, 6],
            "nome": ["seg", "ter", "qua", "qui", "sex", "sáb", "dom"],
            "total": [1000, 1100, 1050, 1080, 1020, 500, 480],
            "positivos": [130, 140, 132, 138, 128, 40, 38],
            "prevalencia": [0.130, 0.127, 0.126, 0.128, 0.125, 0.080, 0.079],
            "ic_inf": [0.110, 0.108, 0.107, 0.109, 0.106, 0.058, 0.057],
            "ic_sup": [0.152, 0.149, 0.148, 0.150, 0.147, 0.108, 0.108],
        }
    )


def test_figura_prevalencia_dia_semana(df_dia_semana_sintetico: pd.DataFrame) -> None:
    figura = figura_prevalencia_dia_semana(df_dia_semana_sintetico)
    assert isinstance(figura, mfigure.Figure)
    (eixo,) = figura.axes
    assert "dia-da-semana" in eixo.get_title().lower()


def test_figura_heatmap_prevalencia_mes_ano() -> None:
    # 2 anos × 3 meses; um mês com total baixo para exercitar o mascaramento.
    dados = pd.DataFrame(
        {
            "ano": [2016, 2016, 2016, 2017, 2017, 2017],
            "mes": [1, 2, 3, 1, 2, 3],
            "total": [1000, 1100, 50, 1200, 1300, 800],
            "positivos": [120, 140, 8, 150, 160, 100],
            "prevalencia": [0.120, 0.127, 0.16, 0.125, 0.123, 0.125],
        }
    )
    figura = figura_heatmap_prevalencia_mes_ano(dados, n_minimo=100)
    assert isinstance(figura, mfigure.Figure)
    eixos = figura.axes
    # Heatmap principal + barra de cor; garantir que há pelo menos um eixo com título.
    titulos = [e.get_title() for e in eixos if e.get_title()]
    assert any("Prevalência" in t for t in titulos)


def test_figura_contagem_diaria_com_anomalias() -> None:
    datas = pd.date_range("2016-01-01", periods=10, freq="D")
    calendario = pd.DataFrame({"data": datas, "total": [100] * 10})
    anomalias = pd.DataFrame(
        {
            "data": [datas[3], datas[7]],
            "total": [1000, 20],
            "z_classico": [5.0, -3.5],
            "z_robusto": [4.5, -1.0],
            "flag_classico": [True, True],
            "flag_robusto": [True, False],
            "flag_ambos": [True, False],
        }
    )
    figura = figura_contagem_diaria_com_anomalias(calendario, anomalias)
    assert isinstance(figura, mfigure.Figure)
    (eixo,) = figura.axes
    assert "anomalias" in eixo.get_title().lower()


def test_figura_contagem_diaria_sem_anomalias() -> None:
    # Contrato: o DataFrame de anomalias pode vir vazio.
    datas = pd.date_range("2016-01-01", periods=5, freq="D")
    calendario = pd.DataFrame({"data": datas, "total": [100] * 5})
    anomalias_vazias = pd.DataFrame(
        {
            "data": pd.to_datetime([]),
            "total": pd.Series([], dtype="int64"),
            "z_classico": pd.Series([], dtype="float64"),
            "z_robusto": pd.Series([], dtype="float64"),
            "flag_classico": pd.Series([], dtype="bool"),
            "flag_robusto": pd.Series([], dtype="bool"),
            "flag_ambos": pd.Series([], dtype="bool"),
        }
    )
    figura = figura_contagem_diaria_com_anomalias(calendario, anomalias_vazias)
    assert isinstance(figura, mfigure.Figure)
