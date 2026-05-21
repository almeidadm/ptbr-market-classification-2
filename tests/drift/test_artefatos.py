"""Testes para `src.drift.artefatos` (ADR 011 §D.6)."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.drift.artefatos import (
    BLOCOS_VALIDOS,
    carregar_json,
    carregar_resultados,
    construir_metadata_drift,
    hash_arquivo,
    montar_dir_drift,
    salvar_json,
    salvar_resultados,
)


# --- montar_dir_drift ---


def test_montar_dir_drift_b1_cria_estrutura(tmp_path: Path) -> None:
    destino = montar_dir_drift(
        "b1_statistical",
        granularidade="mensal",
        escopo="global",
        raiz=tmp_path,
        timestamp="20260520-1200",
    )
    assert destino.exists()
    assert destino.is_dir()
    assert destino.parent.name == "b1_statistical"
    assert destino.name == "20260520-1200-mensal-global"


def test_montar_dir_drift_b3_aceita_sem_granularidade(tmp_path: Path) -> None:
    destino = montar_dir_drift(
        "b3_cpd",
        escopo="mercado",
        raiz=tmp_path,
        timestamp="20260520-1200",
    )
    assert destino.name == "20260520-1200-diario-mercado"


def test_montar_dir_drift_embeddings_sem_timestamp(tmp_path: Path) -> None:
    destino = montar_dir_drift(
        "embeddings",
        escopo="bertimbau_base_cls",
        raiz=tmp_path,
    )
    assert destino == tmp_path / "embeddings" / "bertimbau_base_cls"
    assert destino.exists()


def test_montar_dir_drift_versionamento_v2_em_colisao(tmp_path: Path) -> None:
    primeiro = montar_dir_drift(
        "b2_semantic",
        granularidade="mensal",
        escopo="global",
        raiz=tmp_path,
        timestamp="20260520-1200",
    )
    segundo = montar_dir_drift(
        "b2_semantic",
        granularidade="mensal",
        escopo="global",
        raiz=tmp_path,
        timestamp="20260520-1200",
    )
    terceiro = montar_dir_drift(
        "b2_semantic",
        granularidade="mensal",
        escopo="global",
        raiz=tmp_path,
        timestamp="20260520-1200",
    )
    assert primeiro.name == "20260520-1200-mensal-global"
    assert segundo.name == "20260520-1200-mensal-global-v2"
    assert terceiro.name == "20260520-1200-mensal-global-v3"


def test_montar_dir_drift_rejeita_bloco_desconhecido(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Bloco inválido"):
        montar_dir_drift("blocox", escopo="global", raiz=tmp_path)


def test_montar_dir_drift_b1_exige_granularidade(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="granularidade"):
        montar_dir_drift("b1_statistical", escopo="global", raiz=tmp_path)


def test_montar_dir_drift_b1_exige_escopo(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="escopo"):
        montar_dir_drift("b1_statistical", granularidade="mensal", raiz=tmp_path)


def test_blocos_validos_inclui_todos_esperados() -> None:
    assert set(BLOCOS_VALIDOS) == {
        "b1_statistical",
        "b2_semantic",
        "b3_cpd",
        "embeddings",
    }


# --- construir_metadata_drift ---


def test_construir_metadata_drift_campos_minimos(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.csv"
    corpus.write_text("a,b\n1,2\n", encoding="utf-8")
    metadata = construir_metadata_drift(
        bloco="b1_statistical",
        escopo="global",
        granularidade="mensal",
        corpus=corpus,
        n_artigos=160_000,
        embedding="bertimbau_base_cls",
    )
    assert metadata["adr_referencia"] == "011"
    assert metadata["bloco"] == "b1_statistical"
    assert metadata["granularidade"] == "mensal"
    assert metadata["escopo"] == "global"
    assert metadata["seed_global"] == 2026
    assert metadata["alpha"] == 0.05
    assert metadata["n_repeticoes"] == 5
    assert metadata["embedding"] == "bertimbau_base_cls"
    assert metadata["corpus"]["caminho"] == str(corpus)
    assert metadata["corpus"]["n_artigos"] == 160_000
    assert metadata["corpus"]["sha256"] is not None
    assert metadata["tests"] == ["KS", "CVM", "KTS", "LSDD"]
    assert "framework_versions" in metadata
    assert "python" in metadata["framework_versions"]


def test_construir_metadata_drift_rejeita_bloco_invalido(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.csv"
    corpus.write_text("a\n1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Bloco inválido"):
        construir_metadata_drift(
            bloco="bX",
            escopo="global",
            granularidade="mensal",
            corpus=corpus,
            n_artigos=1,
            embedding="x",
        )


def test_construir_metadata_drift_aceita_extras(tmp_path: Path) -> None:
    corpus = tmp_path / "c.csv"
    corpus.write_text("x\n", encoding="utf-8")
    metadata = construir_metadata_drift(
        bloco="b2_semantic",
        escopo="mercado",
        granularidade="mensal",
        corpus=corpus,
        n_artigos=10,
        embedding="bertimbau_base_cls",
        extras={"observacao": "smoke test"},
    )
    assert metadata["extras"] == {"observacao": "smoke test"}


# --- hash_arquivo ---


def test_hash_arquivo_consistente_para_mesmo_conteudo(tmp_path: Path) -> None:
    a = tmp_path / "a.txt"
    a.write_text("conteudo identico", encoding="utf-8")
    h1 = hash_arquivo(a)
    h2 = hash_arquivo(a)
    assert h1 == h2
    assert isinstance(h1, str)
    assert len(h1) == 64


def test_hash_arquivo_difere_para_conteudo_diferente(tmp_path: Path) -> None:
    a = tmp_path / "a.txt"
    b = tmp_path / "b.txt"
    a.write_text("conteudo A", encoding="utf-8")
    b.write_text("conteudo B", encoding="utf-8")
    assert hash_arquivo(a) != hash_arquivo(b)


def test_hash_arquivo_retorna_none_se_inexistente(tmp_path: Path) -> None:
    assert hash_arquivo(tmp_path / "naoexiste.txt") is None


# --- salvar/carregar JSON e Parquet ---


def test_salvar_e_carregar_json_round_trip(tmp_path: Path) -> None:
    caminho = tmp_path / "meta.json"
    dados = {"a": 1, "b": [1, 2, 3], "c": {"d": "texto com acento ã"}}
    salvar_json(caminho, dados)
    recuperado = carregar_json(caminho)
    assert recuperado == dados


def test_salvar_json_preserva_acentos_sem_escapar(tmp_path: Path) -> None:
    caminho = tmp_path / "meta.json"
    salvar_json(caminho, {"texto": "ação não-mercado"})
    bruto = caminho.read_text(encoding="utf-8")
    assert "ação não-mercado" in bruto
    assert "\\u" not in bruto


def test_salvar_e_carregar_resultados_round_trip(tmp_path: Path) -> None:
    dir_exec = tmp_path / "exec"
    dir_exec.mkdir()
    df = pd.DataFrame(
        {
            "janela_a": ["2015-01", "2015-01"],
            "janela_b": ["2015-02", "2015-02"],
            "teste": ["KS", "CVM"],
            "repeticao": [0, 0],
            "condicao": ["time_ordered", "time_ordered"],
            "p_value": [0.123, 0.456],
            "estatistica": [0.05, 0.06],
        }
    )
    caminho = salvar_resultados(dir_exec, df)
    assert caminho == dir_exec / "results.parquet"
    df_recuperado = carregar_resultados(dir_exec)
    pd.testing.assert_frame_equal(df_recuperado, df)


def test_metadata_serializa_como_json_valido(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.csv"
    corpus.write_text("x\n1\n", encoding="utf-8")
    metadata = construir_metadata_drift(
        bloco="b1_statistical",
        escopo="global",
        granularidade="mensal",
        corpus=corpus,
        n_artigos=1,
        embedding="bertimbau_base_cls",
    )
    caminho = tmp_path / "metadata.json"
    salvar_json(caminho, metadata)
    # Confirma que é JSON válido sem default=str ter quebrado tipos básicos.
    recuperado = json.loads(caminho.read_text(encoding="utf-8"))
    assert recuperado["bloco"] == "b1_statistical"
