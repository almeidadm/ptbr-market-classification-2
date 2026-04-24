"""Testes leves para `src.eda.testes_temporais`."""
from __future__ import annotations

import numpy as np
import pytest

from src.eda.testes_temporais import (
    mann_kendall,
    qui_quadrado_k_por_2,
    z_proporcoes,
)


def test_mann_kendall_tendencia_crescente() -> None:
    serie = np.linspace(0.05, 0.20, num=24)
    resultado = mann_kendall(serie)
    assert resultado.nome == "mann_kendall"
    assert resultado.estatistica == pytest.approx(1.0)
    assert resultado.p_valor < 1e-6
    assert resultado.efeito["delta"] > 0


def test_mann_kendall_tendencia_decrescente() -> None:
    serie = np.linspace(0.20, 0.05, num=24)
    resultado = mann_kendall(serie)
    assert resultado.estatistica == pytest.approx(-1.0)
    assert resultado.efeito["delta"] < 0


def test_mann_kendall_amostra_insuficiente() -> None:
    resultado = mann_kendall([0.1, 0.2])
    assert np.isnan(resultado.estatistica)
    assert resultado.efeito["motivo"] == "amostra insuficiente"


def test_z_proporcoes_detecta_diferenca() -> None:
    resultado = z_proporcoes(n1=1000, k1=150, n2=1000, k2=100)
    assert resultado.nome == "z_proporcoes"
    assert resultado.efeito["diferenca"] == pytest.approx(0.05)
    assert resultado.efeito["risco_relativo"] == pytest.approx(1.5)
    assert resultado.p_valor < 0.01


def test_qui_quadrado_rejeita_homogeneidade() -> None:
    tabela = [[150, 850], [100, 900], [200, 800]]
    resultado = qui_quadrado_k_por_2(tabela)
    assert resultado.efeito["graus_liberdade"] == 2
    assert resultado.p_valor < 0.01
    assert len(resultado.efeito["prevalencias_observadas"]) == 3


def test_qui_quadrado_formato_invalido() -> None:
    with pytest.raises(ValueError):
        qui_quadrado_k_por_2([[10, 20]])
