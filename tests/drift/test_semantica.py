"""Testes para `src.drift.semantica` (B2 das ADRs 010 e 011)."""
from __future__ import annotations

import numpy as np
import pytest

from src.config import SEED
from src.drift.semantica import (
    NOMES_METRICAS_B2,
    ResultadoSemantica,
    aplicar_todas,
    centroide,
    cosseno_centroide_consecutivo,
    cosseno_centroide_cumulativo,
    distancia_cosseno,
    mmd2_consecutivo,
)


try:
    import torch  # noqa: F401

    TORCH_DISPONIVEL = True
except ImportError:
    TORCH_DISPONIVEL = False

requer_torch = pytest.mark.skipif(
    not TORCH_DISPONIVEL,
    reason="MMD via alibi-detect exige torch (rodar no Colab).",
)


def _embeddings_janelas_constantes(
    valores: list[float], n_por_janela: int = 20, d: int = 8
) -> list[np.ndarray]:
    """Embeddings de janela uniformes em torno de um valor distinto por janela."""
    return [
        np.full((n_por_janela, d), v, dtype=np.float32) for v in valores
    ]


def _embeddings_janelas_aleatorias(
    n_janelas: int = 3, n_por_janela: int = 50, d: int = 8, seed: int = 0
) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    return [
        rng.standard_normal((n_por_janela, d)).astype(np.float32)
        for _ in range(n_janelas)
    ]


# --- centroide ---


def test_centroide_e_media_aritmetica() -> None:
    x = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
    np.testing.assert_allclose(centroide(x), [3.0, 4.0])


def test_centroide_erra_em_array_vazio() -> None:
    with pytest.raises(ValueError, match="não-vazio"):
        centroide(np.zeros((0, 8), dtype=np.float32))


def test_centroide_erra_em_array_1d() -> None:
    with pytest.raises(ValueError, match="2D"):
        centroide(np.array([1.0, 2.0, 3.0]))


# --- distancia_cosseno ---


def test_cosseno_identico_e_zero() -> None:
    v = np.array([1.0, 2.0, 3.0])
    assert distancia_cosseno(v, v) == pytest.approx(0.0, abs=1e-12)


def test_cosseno_oposto_e_dois() -> None:
    v = np.array([1.0, 0.0])
    assert distancia_cosseno(v, -v) == pytest.approx(2.0, abs=1e-12)


def test_cosseno_simetrico() -> None:
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([4.0, 5.0, 6.0])
    assert distancia_cosseno(a, b) == pytest.approx(distancia_cosseno(b, a))


def test_cosseno_norma_zero_retorna_um() -> None:
    a = np.array([0.0, 0.0])
    b = np.array([1.0, 2.0])
    assert distancia_cosseno(a, b) == 1.0


# --- cosseno_centroide_consecutivo ---


def test_consecutivo_n_resultados_e_n_janelas_menos_um() -> None:
    embs = _embeddings_janelas_aleatorias(n_janelas=5)
    rotulos = [f"j{i}" for i in range(5)]
    r = cosseno_centroide_consecutivo(embs, rotulos)
    assert len(r) == 4
    assert all(isinstance(item, ResultadoSemantica) for item in r)
    assert all(item.metrica == "cosine_centroid" for item in r)


def test_consecutivo_rotulos_corretos() -> None:
    embs = _embeddings_janelas_aleatorias(n_janelas=3)
    rotulos = ["2015-01", "2015-02", "2015-03"]
    r = cosseno_centroide_consecutivo(embs, rotulos)
    pares = [(item.janela_a, item.janela_b) for item in r]
    assert pares == [("2015-01", "2015-02"), ("2015-02", "2015-03")]


def test_consecutivo_janelas_identicas_dao_zero() -> None:
    embs = _embeddings_janelas_constantes([1.0, 1.0, 1.0])
    rotulos = ["a", "b", "c"]
    r = cosseno_centroide_consecutivo(embs, rotulos)
    for item in r:
        assert item.valor == pytest.approx(0.0, abs=1e-12)


def test_consecutivo_determinismo() -> None:
    embs = _embeddings_janelas_aleatorias(n_janelas=4, seed=42)
    rotulos = [f"j{i}" for i in range(4)]
    r1 = cosseno_centroide_consecutivo(embs, rotulos)
    r2 = cosseno_centroide_consecutivo(embs, rotulos)
    assert r1 == r2


def test_consecutivo_erra_se_paridade_quebra() -> None:
    embs = _embeddings_janelas_aleatorias(n_janelas=3)
    with pytest.raises(ValueError, match="!="):
        cosseno_centroide_consecutivo(embs, ["a", "b"])


def test_consecutivo_erra_com_menos_de_duas_janelas() -> None:
    embs = _embeddings_janelas_aleatorias(n_janelas=1)
    with pytest.raises(ValueError, match="2 janelas"):
        cosseno_centroide_consecutivo(embs, ["unica"])


# --- cosseno_centroide_cumulativo ---


def test_cumulativo_n_resultados_e_n_janelas_menos_um() -> None:
    embs = _embeddings_janelas_aleatorias(n_janelas=5)
    rotulos = [f"j{i}" for i in range(5)]
    r = cosseno_centroide_cumulativo(embs, rotulos)
    assert len(r) == 4
    assert all(item.metrica == "cumulative_cosine" for item in r)


def test_cumulativo_rotulo_a_indica_historico() -> None:
    embs = _embeddings_janelas_aleatorias(n_janelas=4)
    rotulos = ["2015-01", "2015-02", "2015-03", "2015-04"]
    r = cosseno_centroide_cumulativo(embs, rotulos)
    assert r[0].janela_a == "historico_ate_2015-01"
    assert r[0].janela_b == "2015-02"
    assert r[-1].janela_a == "historico_ate_2015-03"
    assert r[-1].janela_b == "2015-04"


def test_cumulativo_janelas_identicas_dao_zero() -> None:
    embs = _embeddings_janelas_constantes([2.5] * 4)
    rotulos = list("abcd")
    r = cosseno_centroide_cumulativo(embs, rotulos)
    for item in r:
        assert item.valor == pytest.approx(0.0, abs=1e-12)


def test_cumulativo_drift_monotonico_e_crescente() -> None:
    # Cada janela é uma constante crescente em todas as dimensões; o
    # centróide cumulativo "história" fica para trás, então a distância
    # de cosseno em si ainda pode ser ~0 (mesma direção). Aqui testamos
    # um cenário com mudança de direção entre janelas.
    embs = [
        np.full((20, 4), 1.0, dtype=np.float32),  # direção (1,1,1,1)
        np.full((20, 4), 1.0, dtype=np.float32),
        np.tile(np.array([[1.0, -1.0, 1.0, -1.0]], dtype=np.float32), (20, 1)),
    ]
    rotulos = ["a", "b", "c"]
    r = cosseno_centroide_cumulativo(embs, rotulos)
    # b ainda mesma direção que história → ~0; c muda → > 0.
    assert r[0].valor == pytest.approx(0.0, abs=1e-6)
    assert r[1].valor > 0.5


# --- MMD² (requer torch) ---


@requer_torch
def test_mmd2_consecutivo_estatistica_finita() -> None:
    embs = _embeddings_janelas_aleatorias(n_janelas=3, seed=0)
    rotulos = [f"j{i}" for i in range(3)]
    r = mmd2_consecutivo(embs, rotulos, seed=SEED, device="cpu")
    assert len(r) == 2
    assert all(item.metrica == "mmd2" for item in r)
    assert all(np.isfinite(item.valor) for item in r)


@requer_torch
def test_mmd2_drift_grande_supera_drift_pequeno() -> None:
    rng = np.random.default_rng(0)
    base = rng.standard_normal((100, 8)).astype(np.float32)
    pequeno = base + 0.05
    grande = base + 2.0
    r_pequeno = mmd2_consecutivo([base, pequeno], ["a", "b"], seed=SEED, device="cpu")
    r_grande = mmd2_consecutivo([base, grande], ["a", "b"], seed=SEED, device="cpu")
    assert r_grande[0].valor > r_pequeno[0].valor


# --- aplicar_todas ---


@requer_torch
def test_aplicar_todas_produz_tres_metricas() -> None:
    embs = _embeddings_janelas_aleatorias(n_janelas=4, seed=0)
    rotulos = [f"j{i}" for i in range(4)]
    r = aplicar_todas(embs, rotulos, seed=SEED, device="cpu")
    metricas = {item.metrica for item in r}
    assert metricas == set(NOMES_METRICAS_B2)
    # cada métrica tem N - 1 = 3 entradas.
    for nome in NOMES_METRICAS_B2:
        assert sum(1 for x in r if x.metrica == nome) == 3
