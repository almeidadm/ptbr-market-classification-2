"""Testes para `src.eda.vocabulario`."""
from __future__ import annotations

import pandas as pd
import pytest

from src.eda.vocabulario import (
    jaccard,
    jaccard_conjuntos,
    psi,
    sobreposicao_top_tokens,
    tokens_top_k_tfidf,
)


def test_jaccard_conjuntos_identicos():
    assert jaccard_conjuntos({"a", "b", "c"}, {"a", "b", "c"}) == 1.0


def test_jaccard_conjuntos_disjuntos():
    assert jaccard_conjuntos({"a", "b"}, {"c", "d"}) == 0.0


def test_jaccard_conjuntos_parcial():
    resultado = jaccard_conjuntos({"a", "b", "c"}, {"b", "c", "d"})
    # interseção {b,c} = 2; união {a,b,c,d} = 4 -> 0,5
    assert resultado == pytest.approx(0.5)


def test_jaccard_conjuntos_ambos_vazios():
    # contrato do módulo: ambos vazios -> 0.0 (não NaN, não erro)
    assert jaccard_conjuntos(set(), set()) == 0.0


def test_jaccard_alias():
    # jaccard_conjuntos deve ser alias semântico de jaccard
    assert jaccard({"a"}, {"a", "b"}) == jaccard_conjuntos({"a"}, {"a", "b"})


def _fixture_minima_com_duas_categorias() -> pd.DataFrame:
    """Dois grupos de textos temáticos — 'saúde' vs 'esporte'.

    Textos propositalmente distintos para garantir top-tokens diferentes.
    """
    linhas_saude = [
        "paciente hospital tratamento médico consulta remédio",
        "vacina imunização sistema saúde público hospital",
        "medicamento farmácia tratamento doença paciente",
        "hospital médico remédio consulta exame clínica",
        "enfermeira médico paciente hospital tratamento ambulância",
        "remédio farmácia paciente consulta exame",
        "doença tratamento hospital médico paciente ambulância",
    ]
    linhas_esporte = [
        "jogador gol campeonato time vitória treinador",
        "estádio torcida time jogador gol partida",
        "treinador time jogador gol partida vitória",
        "campeonato vitória time jogador estádio torcida",
        "gol partida campeonato jogador time treinador",
        "torcida estádio partida time treinador gol",
        "vitória campeonato time gol jogador torcida",
    ]
    linhas = (
        [{"category": "saude", "text": t} for t in linhas_saude]
        + [{"category": "esporte", "text": t} for t in linhas_esporte]
    )
    return pd.DataFrame(linhas)


def test_tokens_top_k_tfidf_separa_categorias():
    df = _fixture_minima_com_duas_categorias()
    saida = tokens_top_k_tfidf(
        df=df,
        coluna_categoria="category",
        coluna_texto="text",
        categorias=["saude", "esporte"],
        k=5,
        n_amostra_por_categoria=20,
        seed=20260423,
    )

    # 2 categorias × 5 = 10 linhas
    assert len(saida) == 10
    assert set(saida["categoria"].unique()) == {"saude", "esporte"}

    tokens_saude = set(saida.loc[saida["categoria"] == "saude", "token"])
    tokens_esporte = set(saida.loc[saida["categoria"] == "esporte", "token"])

    # Tokens temáticos devem estar nos respectivos top-5
    assert any(t in tokens_saude for t in ["hospital", "paciente", "médico", "tratamento"])
    assert any(t in tokens_esporte for t in ["gol", "time", "jogador", "campeonato"])
    # E não devem cruzar
    assert not tokens_saude & {"gol", "time", "jogador"}


def test_tokens_top_k_tfidf_determinista():
    df = _fixture_minima_com_duas_categorias()
    saida1 = tokens_top_k_tfidf(
        df=df,
        coluna_categoria="category",
        coluna_texto="text",
        categorias=["saude", "esporte"],
        k=5,
        n_amostra_por_categoria=5,
        seed=20260423,
    )
    saida2 = tokens_top_k_tfidf(
        df=df,
        coluna_categoria="category",
        coluna_texto="text",
        categorias=["saude", "esporte"],
        k=5,
        n_amostra_por_categoria=5,
        seed=20260423,
    )
    pd.testing.assert_frame_equal(saida1, saida2)


def test_sobreposicao_top_tokens_identifica_compartilhados():
    dados = pd.DataFrame(
        [
            {"categoria": "mercado", "rank": 1, "token": "banco", "tfidf_mean": 0.3},
            {"categoria": "mercado", "rank": 2, "token": "juros", "tfidf_mean": 0.2},
            {"categoria": "mercado", "rank": 3, "token": "bolsa", "tfidf_mean": 0.1},
            {"categoria": "esporte", "rank": 1, "token": "gol", "tfidf_mean": 0.5},
            {"categoria": "esporte", "rank": 2, "token": "banco", "tfidf_mean": 0.1},
            {"categoria": "esporte", "rank": 3, "token": "time", "tfidf_mean": 0.1},
        ]
    )
    resultado = sobreposicao_top_tokens(dados, categoria_referencia="mercado")
    assert len(resultado) == 1
    linha = resultado.iloc[0]
    assert linha["categoria_vs"] == "esporte"
    # interseção {banco} = 1; união {banco, juros, bolsa, gol, time} = 5 -> 0,2
    assert linha["jaccard_top30"] == pytest.approx(0.2)
    assert linha["tokens_compartilhados"] == "banco"
    # exclusivos da referência (mercado): bolsa, juros
    assert set(linha["tokens_exclusivos_referencia"].split(";")) == {"bolsa", "juros"}


def test_psi_zero_para_distribuicoes_iguais():
    dist = {"a": 0.5, "b": 0.3, "c": 0.2}
    assert psi(dist, dist) == pytest.approx(0.0, abs=1e-10)


def test_psi_positivo_para_distribuicoes_diferentes():
    esperado = {"a": 0.5, "b": 0.3, "c": 0.2}
    observado = {"a": 0.2, "b": 0.3, "c": 0.5}
    assert psi(esperado, observado) > 0.0
