"""Testes para `src.drift.testes_estatisticos` (B1 da ADR 010)."""
from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import chi2

from src.config import SEED
from src.drift.testes_estatisticos import (
    Resultado,
    aplicar_todos,
    fisher_combinar,
    aplicar_cvm,
    aplicar_ks,
    aplicar_kts,
    aplicar_lsdd,
)


# Backend pytorch é exigido por MMDDrift e LSDDDrift. Localmente o venv
# pode não ter torch (regra: execução pesada no Colab) — skipamos só os
# testes multivariados, KS/CVM rodam sempre.
try:
    import torch  # noqa: F401

    TORCH_DISPONIVEL = True
except ImportError:
    TORCH_DISPONIVEL = False

requer_torch = pytest.mark.skipif(
    not TORCH_DISPONIVEL,
    reason="MMD/LSDD via alibi-detect exigem torch (rodar no Colab).",
)


def _amostras_identicas(n: int = 200, d: int = 8, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x_a = rng.standard_normal((n, d)).astype(np.float32)
    x_b = rng.standard_normal((n, d)).astype(np.float32)
    return x_a, x_b


def _amostras_com_drift(
    n: int = 200,
    d: int = 8,
    deslocamento: float = 3.0,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x_a = rng.standard_normal((n, d)).astype(np.float32)
    x_b = (rng.standard_normal((n, d)) + deslocamento).astype(np.float32)
    return x_a, x_b


# --- fisher_combinar ---


def test_fisher_p_alto_para_p_values_uniformes() -> None:
    # Sob H0, p-values são uniformes em [0,1]. Fisher de uniformes não
    # acumula evidência contra H0; p combinado tende a ficar acima de
    # 0.01 em amostras razoáveis.
    rng = np.random.default_rng(0)
    p_unif = rng.uniform(0.05, 0.95, size=100)
    p_combinado, chi2_stat = fisher_combinar(p_unif)
    assert 0.0 < p_combinado <= 1.0
    df = 2 * len(p_unif)
    assert pytest.approx(chi2.sf(chi2_stat, df=df), rel=1e-6) == p_combinado


def test_fisher_p_baixo_para_p_values_pequenos() -> None:
    p_baixos = np.full(50, 1e-4)
    p_combinado, _ = fisher_combinar(p_baixos)
    assert p_combinado < 1e-10


def test_fisher_clipa_zero_sem_estourar() -> None:
    p_com_zero = np.array([0.0, 0.5, 0.5])
    p_combinado, chi2_stat = fisher_combinar(p_com_zero)
    assert np.isfinite(chi2_stat)
    assert 0.0 <= p_combinado <= 1.0


def test_fisher_p_values_vazio_erra() -> None:
    with pytest.raises(ValueError):
        fisher_combinar(np.array([]))


# --- KS e CVM (sempre rodam — não dependem de torch) ---


def test_ks_sem_drift_p_value_alto() -> None:
    x_a, x_b = _amostras_identicas(n=300, d=8, seed=0)
    r = aplicar_ks(x_a, x_b)
    assert isinstance(r, Resultado)
    assert 0.0 <= r.p_value <= 1.0
    assert r.p_value > 0.01


def test_ks_com_drift_p_value_baixo() -> None:
    x_a, x_b = _amostras_com_drift(n=300, d=8, deslocamento=3.0, seed=0)
    r = aplicar_ks(x_a, x_b)
    assert r.p_value < 1e-10


def test_cvm_sem_drift_p_value_alto() -> None:
    x_a, x_b = _amostras_identicas(n=300, d=8, seed=0)
    r = aplicar_cvm(x_a, x_b)
    assert r.p_value > 0.01


def test_cvm_com_drift_p_value_baixo() -> None:
    x_a, x_b = _amostras_com_drift(n=300, d=8, deslocamento=3.0, seed=0)
    r = aplicar_cvm(x_a, x_b)
    assert r.p_value < 1e-10


def test_ks_determinismo() -> None:
    x_a, x_b = _amostras_identicas(seed=0)
    r1 = aplicar_ks(x_a, x_b)
    r2 = aplicar_ks(x_a, x_b)
    assert r1 == r2


def test_cvm_determinismo() -> None:
    x_a, x_b = _amostras_identicas(seed=0)
    r1 = aplicar_cvm(x_a, x_b)
    r2 = aplicar_cvm(x_a, x_b)
    assert r1 == r2


# --- KTS e LSDD (skipados sem torch) ---


@requer_torch
def test_kts_sem_drift_p_value_alto() -> None:
    x_a, x_b = _amostras_identicas(n=200, d=8, seed=0)
    r = aplicar_kts(x_a, x_b, n_permutations=50, seed=SEED, device="cpu")
    assert 0.0 <= r.p_value <= 1.0
    assert r.p_value > 0.05


@requer_torch
def test_kts_com_drift_p_value_baixo() -> None:
    x_a, x_b = _amostras_com_drift(n=200, d=8, deslocamento=2.0, seed=0)
    r = aplicar_kts(x_a, x_b, n_permutations=50, seed=SEED, device="cpu")
    assert r.p_value < 0.05


@requer_torch
def test_lsdd_sem_drift_p_value_alto() -> None:
    x_a, x_b = _amostras_identicas(n=200, d=8, seed=0)
    r = aplicar_lsdd(x_a, x_b, n_permutations=50, seed=SEED, device="cpu")
    assert 0.0 <= r.p_value <= 1.0
    assert r.p_value > 0.05


@requer_torch
def test_lsdd_com_drift_p_value_baixo() -> None:
    x_a, x_b = _amostras_com_drift(n=200, d=8, deslocamento=2.0, seed=0)
    r = aplicar_lsdd(x_a, x_b, n_permutations=50, seed=SEED, device="cpu")
    assert r.p_value < 0.05


@requer_torch
def test_kts_determinismo_sob_mesma_seed() -> None:
    x_a, x_b = _amostras_com_drift(n=200, d=8, seed=0)
    r1 = aplicar_kts(x_a, x_b, n_permutations=50, seed=SEED, device="cpu")
    r2 = aplicar_kts(x_a, x_b, n_permutations=50, seed=SEED, device="cpu")
    # Statistic é determinística; p-value pode ter pequena flutuação por
    # rng do alibi-detect, mas deve ser muito próximo.
    assert r1.estatistica == pytest.approx(r2.estatistica)
    assert r1.p_value == pytest.approx(r2.p_value, abs=0.02)


# --- aplicar_todos ---


@requer_torch
def test_aplicar_todos_retorna_quatro_testes() -> None:
    x_a, x_b = _amostras_identicas(n=200, d=8, seed=0)
    resultados = aplicar_todos(
        x_a, x_b, n_permutations=50, seed=SEED, device="cpu"
    )
    assert set(resultados.keys()) == {"KS", "CVM", "KTS", "LSDD"}
    for nome, r in resultados.items():
        assert isinstance(r, Resultado), f"{nome} não devolveu Resultado"
        assert 0.0 <= r.p_value <= 1.0
        assert np.isfinite(r.estatistica)
