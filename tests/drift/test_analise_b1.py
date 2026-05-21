"""Testes para `src.drift.analise_b1`."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.drift.analise_b1 import (
    agregar_tabela_wanderley,
    descobrir_runs,
    eh_smoke_test,
    escolher_corrida_principal_por_combo,
    renderizar_tabela_markdown,
    resumir_achados,
)
from src.drift.artefatos import construir_metadata_drift, salvar_json, salvar_resultados


NOMES_TESTES = ("KS", "CVM", "KTS", "LSDD")


def _fabricar_resultados(
    n_pares: int,
    p_value_time: float,
    p_value_rand: float,
    n_repeticoes: int = 5,
) -> pd.DataFrame:
    """Constrói um DataFrame realista para um combo."""
    linhas = []
    for i in range(n_pares):
        for teste in NOMES_TESTES:
            linhas.append(
                {
                    "janela_a": f"{2015 + i // 12:04d}-{(i % 12) + 1:02d}",
                    "janela_b": f"{2015 + (i + 1) // 12:04d}-{((i + 1) % 12) + 1:02d}",
                    "teste": teste,
                    "repeticao": 0,
                    "condicao": "time_ordered",
                    "p_value": p_value_time,
                    "estatistica": 100.0,
                }
            )
        for rep in range(n_repeticoes):
            for teste in NOMES_TESTES:
                linhas.append(
                    {
                        "janela_a": f"rand_{i:03d}",
                        "janela_b": f"rand_{i + 1:03d}",
                        "teste": teste,
                        "repeticao": rep,
                        "condicao": "randomized",
                        "p_value": p_value_rand,
                        "estatistica": 10.0,
                    }
                )
    return pd.DataFrame(linhas)


def _gravar_run(
    raiz: Path,
    *,
    granularidade: str,
    escopo: str,
    df: pd.DataFrame,
    limite_pares: int | None = None,
    timestamp: str = "20260521-0719",
) -> Path:
    dir_run = raiz / f"{timestamp}-{granularidade}-{escopo}"
    dir_run.mkdir(parents=True)
    salvar_resultados(dir_run, df)
    corpus_fake = raiz / "corpus_fake.parquet"
    corpus_fake.write_bytes(b"fake")
    metadata = construir_metadata_drift(
        bloco="b1_statistical",
        escopo=escopo,
        granularidade=granularidade,
        corpus=corpus_fake,
        n_artigos=100,
        embedding="bertimbau_base_cls",
        extras={"limite_pares": limite_pares},
    )
    salvar_json(dir_run / "metadata.json", metadata)
    return dir_run


# --- eh_smoke_test ---


def test_eh_smoke_test_com_limite() -> None:
    assert eh_smoke_test({"extras": {"limite_pares": 2}})


def test_eh_smoke_test_sem_limite() -> None:
    assert not eh_smoke_test({"extras": {"limite_pares": None}})


def test_eh_smoke_test_sem_extras() -> None:
    assert not eh_smoke_test({})


# --- descobrir_runs ---


def test_descobrir_runs_em_diretorio_inexistente(tmp_path: Path) -> None:
    assert descobrir_runs(tmp_path / "nao-existe") == []


def test_descobrir_runs_ignora_diretorios_com_underscore(tmp_path: Path) -> None:
    (tmp_path / "20260101-foo").mkdir()
    (tmp_path / "_auxiliar").mkdir()
    (tmp_path / "arquivo.txt").write_text("ignorar")
    encontrados = descobrir_runs(tmp_path)
    assert len(encontrados) == 1
    assert encontrados[0].name == "20260101-foo"


# --- escolher_corrida_principal_por_combo ---


def test_escolher_descarta_smoke_e_pega_mais_completo(tmp_path: Path) -> None:
    df_completo = _fabricar_resultados(n_pares=33, p_value_time=0.0, p_value_rand=0.5)
    df_smoke = _fabricar_resultados(n_pares=1, p_value_time=0.0, p_value_rand=0.5)

    _gravar_run(
        tmp_path,
        granularidade="mensal",
        escopo="global",
        df=df_completo,
        limite_pares=None,
        timestamp="20260521-0719",
    )
    _gravar_run(
        tmp_path,
        granularidade="mensal",
        escopo="global",
        df=df_smoke,
        limite_pares=1,
        timestamp="20260521-0823",
    )

    from src.drift.analise_b1 import carregar_run

    runs = []
    for d in descobrir_runs(tmp_path):
        carregado = carregar_run(d)
        if carregado is not None:
            metadata, df = carregado
            runs.append((d, metadata, df))

    escolhidos = escolher_corrida_principal_por_combo(runs)
    assert ("mensal", "global") in escolhidos
    _, _, df_escolhido = escolhidos[("mensal", "global")]
    # 33 pares × 4 testes + 33 pares × 5 reps × 4 testes = 132 + 660 = 792
    assert len(df_escolhido) == 792


def test_escolher_corrida_principal_dois_combos_distintos(tmp_path: Path) -> None:
    df1 = _fabricar_resultados(n_pares=10, p_value_time=0.0, p_value_rand=0.5)
    df2 = _fabricar_resultados(n_pares=10, p_value_time=0.0, p_value_rand=0.5)
    _gravar_run(
        tmp_path,
        granularidade="mensal",
        escopo="global",
        df=df1,
        timestamp="20260521-0719",
    )
    _gravar_run(
        tmp_path,
        granularidade="mensal",
        escopo="mercado",
        df=df2,
        timestamp="20260521-0729",
    )

    from src.drift.analise_b1 import carregar_run

    runs = []
    for d in descobrir_runs(tmp_path):
        carregado = carregar_run(d)
        if carregado is not None:
            metadata, df = carregado
            runs.append((d, metadata, df))

    escolhidos = escolher_corrida_principal_por_combo(runs)
    assert set(escolhidos.keys()) == {("mensal", "global"), ("mensal", "mercado")}


# --- agregar_tabela_wanderley ---


def test_agregar_produz_uma_linha_por_combinacao(tmp_path: Path) -> None:
    df = _fabricar_resultados(n_pares=5, p_value_time=0.0, p_value_rand=0.5)
    dir_run = _gravar_run(
        tmp_path, granularidade="mensal", escopo="global", df=df
    )
    escolhidos = {
        ("mensal", "global"): (dir_run, {}, df),
    }
    tabela = agregar_tabela_wanderley(escolhidos)
    # 4 testes × 2 condições = 8 linhas
    assert len(tabela) == 8
    assert set(tabela["teste"]) == set(NOMES_TESTES)
    assert set(tabela["condicao"]) == {"time_ordered", "randomized"}


def test_agregar_mean_bate_com_dados_fabricados() -> None:
    df = _fabricar_resultados(n_pares=10, p_value_time=0.01, p_value_rand=0.5)
    escolhidos = {("mensal", "global"): (Path("dummy"), {}, df)}
    tabela = agregar_tabela_wanderley(escolhidos)

    for _, r in tabela.iterrows():
        esperado = 0.01 if r["condicao"] == "time_ordered" else 0.5
        assert r["mean"] == pytest.approx(esperado, abs=1e-9)
        assert r["std"] == pytest.approx(0.0, abs=1e-9)


def test_agregar_n_correto_por_condicao() -> None:
    df = _fabricar_resultados(n_pares=33, p_value_time=0.0, p_value_rand=0.5, n_repeticoes=5)
    escolhidos = {("mensal", "global"): (Path("dummy"), {}, df)}
    tabela = agregar_tabela_wanderley(escolhidos)

    n_time = tabela[tabela["condicao"] == "time_ordered"]["n"].iloc[0]
    n_rand = tabela[tabela["condicao"] == "randomized"]["n"].iloc[0]
    assert n_time == 33
    assert n_rand == 33 * 5


# --- renderizar_tabela_markdown ---


def test_renderizar_markdown_tabela_vazia() -> None:
    md = renderizar_tabela_markdown(pd.DataFrame())
    assert "(nenhum dado)" in md


def test_renderizar_markdown_inclui_todos_combos() -> None:
    df = _fabricar_resultados(n_pares=3, p_value_time=0.0, p_value_rand=0.5)
    escolhidos = {
        ("mensal", "global"): (Path("d1"), {}, df),
        ("bisemanal", "mercado"): (Path("d2"), {}, df),
    }
    tabela = agregar_tabela_wanderley(escolhidos)
    md = renderizar_tabela_markdown(tabela)
    assert "global — mensal" in md
    assert "mercado — bisemanal" in md
    for teste in NOMES_TESTES:
        assert f"| {teste} |" in md


# --- resumir_achados ---


def test_resumir_achados_inclui_gap_positivo() -> None:
    df = _fabricar_resultados(n_pares=5, p_value_time=0.0, p_value_rand=0.5)
    escolhidos = {("mensal", "global"): (Path("d"), {}, df)}
    tabela = agregar_tabela_wanderley(escolhidos)
    md = resumir_achados(tabela)
    assert "global" in md
    assert "+0.500" in md  # gap = randomized - time_ordered = 0.5 - 0 = 0.5
