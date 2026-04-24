"""Testes para `src.experimento.artefatos`."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.experimento.artefatos import (
    carregar_best_hp_do_primario,
    construir_metadata,
    montar_dir_experimento,
    salvar_hp_search,
    salvar_json,
    salvar_predictions,
)


def test_montar_dir_cria_estrutura_esperada(tmp_path: Path) -> None:
    dir_exp = montar_dir_experimento(
        "logreg", "opcao7", "rolling", "multi", raiz=tmp_path
    )
    assert dir_exp.exists()
    assert (dir_exp / "folds").exists()
    assert dir_exp.name.endswith("-logreg-opcao7-rolling-multi")


def test_montar_dir_versiona_ao_colidir(tmp_path: Path) -> None:
    dir1 = montar_dir_experimento(
        "logreg", "opcao7", "rolling", "multi", raiz=tmp_path, timestamp="20260424-1200"
    )
    dir2 = montar_dir_experimento(
        "logreg", "opcao7", "rolling", "multi", raiz=tmp_path, timestamp="20260424-1200"
    )
    assert dir1 != dir2
    assert dir2.name.endswith("-v2")


def test_salvar_json_e_ler_de_volta(tmp_path: Path) -> None:
    caminho = tmp_path / "a" / "b.json"
    salvar_json(caminho, {"x": 1, "y": None, "lista": [1, 2, 3]})
    lido = json.loads(caminho.read_text(encoding="utf-8"))
    assert lido == {"x": 1, "y": None, "lista": [1, 2, 3]}


def test_salvar_predictions_inclui_scores_por_classe(tmp_path: Path) -> None:
    dir_fold = tmp_path / "fold_0"
    dir_fold.mkdir()
    df_teste = pd.DataFrame(
        {
            "link": ["a", "b", "c"],
            "date": pd.to_datetime(["2016-07-01", "2016-07-02", "2016-07-03"]),
            "y_original": ["mercado", "poder", "outros"],
            "y_colapsado": ["mercado", "poder", "outros"],
            "eh_duplicata_text": [False, False, False],
        }
    )
    y_pred = np.array(["mercado", "poder", "outros"])
    scores = np.array([[0.9, 0.05, 0.05], [0.1, 0.8, 0.1], [0.3, 0.3, 0.4]])
    classes = ["mercado", "poder", "outros"]
    caminho = salvar_predictions(dir_fold, df_teste, y_pred, scores, classes)

    df_persistido = pd.read_parquet(caminho)
    assert {"link", "date", "y_original", "y_colapsado", "y_pred", "eh_duplicata_text"} <= set(
        df_persistido.columns
    )
    assert {"score_mercado", "score_poder", "score_outros"} <= set(df_persistido.columns)
    assert df_persistido["y_pred"].tolist() == ["mercado", "poder", "outros"]


def test_salvar_hp_search_como_csv(tmp_path: Path) -> None:
    dir_fold = tmp_path / "fold_0"
    dir_fold.mkdir()
    tabela = pd.DataFrame(
        [
            {"C": 0.1, "class_weight": None, "f1_binario_inner_val": 0.70},
            {"C": 1.0, "class_weight": None, "f1_binario_inner_val": 0.75},
        ]
    )
    caminho = salvar_hp_search(dir_fold, tabela)
    persistida = pd.read_csv(caminho)
    assert len(persistida) == 2


def test_carregar_best_hp_do_primario(tmp_path: Path) -> None:
    dir_fold = tmp_path / "folds" / "fold_0"
    dir_fold.mkdir(parents=True)
    salvar_json(dir_fold / "best_hp.json", {"C": 1.0, "class_weight": "balanced"})
    hp = carregar_best_hp_do_primario(tmp_path, fold_idx=0)
    assert hp == {"C": 1.0, "class_weight": "balanced"}


def test_construir_metadata_campos_minimos(tmp_path: Path) -> None:
    corpus_raw = tmp_path / "raw.csv"
    corpus_raw.write_text("link,title\nx,y\n")
    corpus_proc = tmp_path / "proc.parquet"
    corpus_proc.write_bytes(b"fake")

    metadata = construir_metadata(
        familia="logreg",
        recorte="opcao7",
        protocolo="rolling",
        regime="multi",
        n_classes_treino=8,
        n_folds=5,
        hp_search_space={"grid": [{"C": 1.0}]},
        hp_por_fold=[{"fold": 0, "hp": {"C": 1.0}}],
        corpus_raw=corpus_raw,
        corpus_processado=corpus_proc,
        n_artigos=1000,
    )
    for campo in (
        "experiment_id",
        "timestamp_iso",
        "seed",
        "familia_modelo",
        "recorte",
        "protocolo_split",
        "regime",
        "corpus",
        "hp_search_space",
        "hp_per_fold",
        "framework_versions",
    ):
        assert campo in metadata
    assert metadata["seed"] == 2026
    assert metadata["corpus"]["n_artigos"] == 1000
    assert metadata["corpus"]["raw_sha256"] is not None
