"""Testes leves para `src.eda.anomalias`."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.eda.anomalias import (
    detectar_anomalias_diarias,
    detectar_dias_zero,
    gerar_calendario,
)


def test_gerar_calendario_preenche_dias_faltantes() -> None:
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2016-01-01",
                    "2016-01-01",
                    "2016-01-03",
                    "2016-01-05",
                ]
            )
        }
    )
    calendario = gerar_calendario(df)

    assert list(calendario.columns) == ["data", "total"]
    # 5 dias entre min e max, com dois deles zerados (02 e 04).
    assert len(calendario) == 5
    assert calendario["total"].tolist() == [2, 0, 1, 0, 1]


def test_detectar_dias_zero_lista_somente_zeros() -> None:
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2016-01-01", "2016-01-03"]
            )
        }
    )
    calendario = gerar_calendario(df)
    zeros = detectar_dias_zero(calendario)

    assert len(zeros) == 1
    assert zeros["data"].iloc[0] == pd.Timestamp("2016-01-02")
    assert zeros["dia_semana"].iloc[0] in {"seg", "ter", "qua", "qui", "sex", "sáb", "dom"}


def test_detectar_anomalias_diarias_captura_outlier_plantado() -> None:
    # 30 dias com contagem em torno de 100 (com pequeno ruído determinístico),
    # e um outlier de 1000 no dia 15. O outlier tem que ser flagado pelas duas
    # regras com limiar 3.
    rng = np.random.default_rng(20260423)
    base = rng.integers(low=95, high=106, size=30)
    base[15] = 1000
    datas = pd.date_range("2016-01-01", periods=30, freq="D")
    calendario = pd.DataFrame({"data": datas, "total": base.astype("int64")})

    anomalias = detectar_anomalias_diarias(calendario, limiar=3.0)
    assert not anomalias.empty
    # O outlier deve estar presente e flagado em ambas as regras.
    linha_outlier = anomalias.loc[anomalias["data"] == pd.Timestamp("2016-01-16")]
    assert len(linha_outlier) == 1
    assert bool(linha_outlier["flag_classico"].iat[0])
    assert bool(linha_outlier["flag_robusto"].iat[0])
    assert bool(linha_outlier["flag_ambos"].iat[0])


def test_detectar_anomalias_ignora_dias_zero_no_calculo() -> None:
    # Série constante (todos 100) com um dia zero — o dia zero está fora do
    # cálculo e não é flagado como anomalia pelo módulo (fica em
    # `detectar_dias_zero`).
    datas = pd.date_range("2016-01-01", periods=30, freq="D")
    totais = np.full(30, 100, dtype="int64")
    totais[10] = 0
    calendario = pd.DataFrame({"data": datas, "total": totais})

    zeros = detectar_dias_zero(calendario)
    anomalias = detectar_anomalias_diarias(calendario, limiar=3.0)

    assert len(zeros) == 1
    assert zeros["data"].iloc[0] == pd.Timestamp("2016-01-11")
    # Nenhuma anomalia sobre os dias que publicaram (série constante).
    assert anomalias.empty


def test_detectar_anomalias_colunas_esperadas() -> None:
    # Mesmo em calendário pequeno o retorno tem contrato estável.
    datas = pd.date_range("2016-01-01", periods=5, freq="D")
    calendario = pd.DataFrame({"data": datas, "total": [100, 100, 100, 1000, 100]})
    anomalias = detectar_anomalias_diarias(calendario, limiar=3.0)

    colunas_esperadas = [
        "data",
        "total",
        "z_classico",
        "z_robusto",
        "flag_classico",
        "flag_robusto",
        "flag_ambos",
    ]
    assert list(anomalias.columns) == colunas_esperadas
