"""Testes para `src.drift.janelas` (ADRs 010 e 011)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import SEED
from src.drift.janelas import (
    Janela,
    gerar_janelas_bisemanais,
    gerar_janelas_mensais,
    gerar_particoes_aleatorizadas,
    gerar_repeticoes_aleatorizadas,
)


def _corpus_diario(
    data_inicio: str,
    n_dias: int,
    por_dia: int = 3,
) -> pd.DataFrame:
    linhas = []
    inicio = pd.Timestamp(data_inicio)
    for d in range(n_dias):
        dia = inicio + pd.Timedelta(days=d)
        for k in range(por_dia):
            linhas.append({"date": dia, "link": f"d{d:03d}-k{k}"})
    return pd.DataFrame(linhas)


# --- Janelas mensais ---


def test_janelas_mensais_cobrem_meses_distintos() -> None:
    df = _corpus_diario("2015-01-01", n_dias=90, por_dia=2)
    janelas = gerar_janelas_mensais(df)
    rotulos = [j.rotulo for j in janelas]
    assert rotulos == ["2015-01", "2015-02", "2015-03"]


def test_janelas_mensais_indices_disjuntos_e_cobrem_corpus_integralmente() -> None:
    df = _corpus_diario("2015-01-01", n_dias=90, por_dia=2)
    janelas = gerar_janelas_mensais(df)
    todos = np.sort(np.concatenate([j.indices_no_corpus for j in janelas]))
    np.testing.assert_array_equal(todos, np.arange(len(df)))


def test_janelas_mensais_inicio_e_fim_alinhados_ao_mes() -> None:
    df = _corpus_diario("2015-01-15", n_dias=60, por_dia=1)
    janelas = gerar_janelas_mensais(df)
    assert janelas[0].inicio == pd.Timestamp("2015-01-01")
    assert janelas[0].fim == pd.Timestamp("2015-02-01")
    assert janelas[1].inicio == pd.Timestamp("2015-02-01")


def test_janelas_mensais_meses_vazios_sao_omitidos() -> None:
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2015-01-15",
                    "2015-01-20",
                    "2015-03-05",  # fevereiro inteiro vazio
                ]
            ),
            "link": ["a", "b", "c"],
        }
    )
    janelas = gerar_janelas_mensais(df)
    rotulos = [j.rotulo for j in janelas]
    assert rotulos == ["2015-01", "2015-03"]


def test_janelas_mensais_determinismo() -> None:
    df = _corpus_diario("2015-01-01", n_dias=90, por_dia=2)
    j1 = gerar_janelas_mensais(df)
    j2 = gerar_janelas_mensais(df)
    for a, b in zip(j1, j2):
        np.testing.assert_array_equal(a.indices_no_corpus, b.indices_no_corpus)


def test_janelas_mensais_erra_se_falta_coluna_date() -> None:
    df = pd.DataFrame({"link": ["a", "b"]})
    with pytest.raises(ValueError, match="`date`"):
        gerar_janelas_mensais(df)


def test_janelas_mensais_erra_se_date_tem_nulo() -> None:
    df = pd.DataFrame(
        {"date": [pd.Timestamp("2015-01-01"), pd.NaT], "link": ["a", "b"]}
    )
    with pytest.raises(ValueError, match="nulos"):
        gerar_janelas_mensais(df)


# --- Janelas bi-semanais ---


def test_janelas_bisemanais_passo_de_14_dias() -> None:
    df = _corpus_diario("2015-01-01", n_dias=60, por_dia=1)
    janelas = gerar_janelas_bisemanais(df)
    for j in janelas:
        assert (j.fim - j.inicio) == pd.Timedelta(days=14)


def test_janelas_bisemanais_descartam_ultima_parcial() -> None:
    # 30 dias começando em 01/01: janelas 01-14, 15-28; 29-30 é parcial.
    df = _corpus_diario("2015-01-01", n_dias=30, por_dia=1)
    janelas = gerar_janelas_bisemanais(df)
    assert len(janelas) == 2
    assert janelas[-1].fim == pd.Timestamp("2015-01-29")


def test_janelas_bisemanais_rotulos_contem_intervalo() -> None:
    df = _corpus_diario("2015-01-01", n_dias=30, por_dia=1)
    janelas = gerar_janelas_bisemanais(df)
    assert janelas[0].rotulo == "2015-01-01_2015-01-14"
    assert janelas[1].rotulo == "2015-01-15_2015-01-28"


def test_janelas_bisemanais_dias_por_janela_nao_positivo_erra() -> None:
    df = _corpus_diario("2015-01-01", n_dias=30, por_dia=1)
    with pytest.raises(ValueError, match="positivo"):
        gerar_janelas_bisemanais(df, dias_por_janela=0)


# --- Partições aleatorizadas ---


def test_particoes_aleatorizadas_respeitam_tamanhos() -> None:
    particoes = gerar_particoes_aleatorizadas(
        n_total=100, tamanhos=[10, 20, 30], seed=SEED
    )
    assert [len(p.indices_no_corpus) for p in particoes] == [10, 20, 30]


def test_particoes_aleatorizadas_sao_disjuntas() -> None:
    particoes = gerar_particoes_aleatorizadas(
        n_total=100, tamanhos=[25, 25, 25, 25], seed=SEED
    )
    todos = np.concatenate([p.indices_no_corpus for p in particoes])
    assert len(set(todos.tolist())) == len(todos)


def test_particoes_aleatorizadas_determinismo_sob_mesma_seed() -> None:
    a = gerar_particoes_aleatorizadas(n_total=200, tamanhos=[50, 50, 50], seed=SEED)
    b = gerar_particoes_aleatorizadas(n_total=200, tamanhos=[50, 50, 50], seed=SEED)
    for pa, pb in zip(a, b):
        np.testing.assert_array_equal(pa.indices_no_corpus, pb.indices_no_corpus)


def test_particoes_aleatorizadas_mudam_com_seed_diferente() -> None:
    a = gerar_particoes_aleatorizadas(n_total=200, tamanhos=[50, 50, 50], seed=SEED)
    b = gerar_particoes_aleatorizadas(
        n_total=200, tamanhos=[50, 50, 50], seed=SEED + 1
    )
    iguais = all(
        np.array_equal(pa.indices_no_corpus, pb.indices_no_corpus)
        for pa, pb in zip(a, b)
    )
    assert not iguais


def test_particoes_aleatorizadas_rotulos_no_padrao_rand() -> None:
    particoes = gerar_particoes_aleatorizadas(
        n_total=100, tamanhos=[10, 10], seed=SEED
    )
    assert all(p.rotulo.startswith("rand_") for p in particoes)
    assert particoes[0].rotulo == "rand_000"
    assert particoes[1].rotulo == "rand_001"


def test_particoes_aleatorizadas_erram_se_soma_excede_total() -> None:
    with pytest.raises(ValueError, match="excede"):
        gerar_particoes_aleatorizadas(n_total=50, tamanhos=[30, 30], seed=SEED)


def test_particoes_aleatorizadas_aceitam_soma_menor_que_total() -> None:
    particoes = gerar_particoes_aleatorizadas(
        n_total=100, tamanhos=[10, 20], seed=SEED
    )
    assert sum(len(p.indices_no_corpus) for p in particoes) == 30


# --- Repetições do baseline ---


def test_repeticoes_geram_n_listas_independentes() -> None:
    df = _corpus_diario("2015-01-01", n_dias=90, por_dia=2)
    janelas = gerar_janelas_mensais(df)
    repeticoes = gerar_repeticoes_aleatorizadas(
        n_total=len(df), janelas_referencia=janelas, seed_global=SEED, n_repeticoes=3
    )
    assert len(repeticoes) == 3
    # Cada repetição tem o mesmo número de partições que janelas_referencia.
    for rep in repeticoes:
        assert len(rep) == len(janelas)


def test_repeticoes_tamanhos_iguais_aos_das_janelas_referencia() -> None:
    df = _corpus_diario("2015-01-01", n_dias=90, por_dia=2)
    janelas = gerar_janelas_mensais(df)
    tamanhos_referencia = [len(j.indices_no_corpus) for j in janelas]
    repeticoes = gerar_repeticoes_aleatorizadas(
        n_total=len(df), janelas_referencia=janelas, seed_global=SEED, n_repeticoes=2
    )
    for rep in repeticoes:
        assert [len(p.indices_no_corpus) for p in rep] == tamanhos_referencia


def test_repeticoes_seeds_consecutivas_produzem_resultados_distintos() -> None:
    df = _corpus_diario("2015-01-01", n_dias=90, por_dia=2)
    janelas = gerar_janelas_mensais(df)
    repeticoes = gerar_repeticoes_aleatorizadas(
        n_total=len(df), janelas_referencia=janelas, seed_global=SEED, n_repeticoes=2
    )
    iguais = all(
        np.array_equal(a.indices_no_corpus, b.indices_no_corpus)
        for a, b in zip(repeticoes[0], repeticoes[1])
    )
    assert not iguais


def test_repeticoes_determinismo_sob_mesma_seed_global() -> None:
    df = _corpus_diario("2015-01-01", n_dias=90, por_dia=2)
    janelas = gerar_janelas_mensais(df)
    r1 = gerar_repeticoes_aleatorizadas(
        n_total=len(df), janelas_referencia=janelas, seed_global=SEED, n_repeticoes=3
    )
    r2 = gerar_repeticoes_aleatorizadas(
        n_total=len(df), janelas_referencia=janelas, seed_global=SEED, n_repeticoes=3
    )
    for rep_a, rep_b in zip(r1, r2):
        for pa, pb in zip(rep_a, rep_b):
            np.testing.assert_array_equal(pa.indices_no_corpus, pb.indices_no_corpus)


# --- Janela.resumir ---


def test_janela_resumir_serializa_campos() -> None:
    janela = Janela(
        indice=3,
        rotulo="2015-04",
        indices_no_corpus=np.array([0, 1, 2, 3]),
        inicio=pd.Timestamp("2015-04-01"),
        fim=pd.Timestamp("2015-05-01"),
    )
    resumo = janela.resumir()
    assert resumo == {
        "indice": 3,
        "rotulo": "2015-04",
        "n_artigos": 4,
        "inicio": "2015-04-01T00:00:00",
        "fim": "2015-05-01T00:00:00",
    }


def test_janela_resumir_com_inicio_e_fim_none() -> None:
    janela = Janela(
        indice=0,
        rotulo="rand_000",
        indices_no_corpus=np.array([0, 1]),
        inicio=None,
        fim=None,
    )
    resumo = janela.resumir()
    assert resumo["inicio"] is None
    assert resumo["fim"] is None
