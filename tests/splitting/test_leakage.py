"""Testes para `src.splitting.leakage` (ADR 008 diagnóstico)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.splitting.fold import Fold
from src.splitting.leakage import diagnosticar_leakage


def _df(ids_raw: list[int]) -> pd.DataFrame:
    return pd.DataFrame({"id_raw": ids_raw, "link": [f"l{i}" for i in ids_raw]})


def _fold_simples(treino: list[int], teste: list[int]) -> Fold:
    return Fold(
        indice=0,
        treino_idx=np.array(treino),
        teste_idx=np.array(teste),
        inner_train_idx=np.array(treino),
        inner_val_idx=np.array([]),
        inicio_teste=None,
        fim_teste=None,
    )


def test_par_cruzando_treino_teste_e_contado_como_vazamento() -> None:
    df = _df([0, 1, 2, 3])
    fold = _fold_simples(treino=[0, 1], teste=[2, 3])
    pares = pd.DataFrame({"id_a": [0], "id_b": [2]})

    resultado = diagnosticar_leakage([fold], df, pares)
    assert resultado["por_fold"][0]["pares_cruzando_treino_teste"] == 1
    assert resultado["por_fold"][0]["pares_presentes"] == 1


def test_par_totalmente_no_treino_nao_e_vazamento() -> None:
    df = _df([0, 1, 2, 3])
    fold = _fold_simples(treino=[0, 1], teste=[2, 3])
    pares = pd.DataFrame({"id_a": [0], "id_b": [1]})

    resultado = diagnosticar_leakage([fold], df, pares)
    assert resultado["por_fold"][0]["pares_cruzando_treino_teste"] == 0
    assert resultado["por_fold"][0]["pares_dentro_treino"] == 1


def test_par_com_um_lado_fora_do_fold_e_descartado() -> None:
    df = _df([0, 1, 2, 3])
    # Fold cobre apenas ids 0, 1, 2; id 3 não está nem em treino nem em teste.
    fold = _fold_simples(treino=[0, 1], teste=[2])
    pares = pd.DataFrame({"id_a": [0], "id_b": [3]})

    resultado = diagnosticar_leakage([fold], df, pares)
    assert resultado["por_fold"][0]["pares_presentes"] == 0


def test_par_com_id_fora_do_corpus_e_descartado_globalmente() -> None:
    df = _df([0, 1, 2])
    fold = _fold_simples(treino=[0], teste=[1, 2])
    pares = pd.DataFrame({"id_a": [0, 99], "id_b": [1, 1]})  # 99 não existe no corpus

    resultado = diagnosticar_leakage([fold], df, pares)
    assert resultado["n_pares_validos"] == 1
    assert resultado["n_pares_csv_bruto"] == 2


def test_agregacao_cross_fold() -> None:
    df = _df([0, 1, 2, 3, 4, 5])
    fold0 = _fold_simples(treino=[0, 1, 2], teste=[3])
    fold1 = _fold_simples(treino=[0, 1, 2, 3], teste=[4, 5])
    pares = pd.DataFrame(
        {
            "id_a": [0, 3, 4],
            "id_b": [3, 4, 5],
        }
    )

    resultado = diagnosticar_leakage([fold0, fold1], df, pares)
    # fold0: par (0,3) cruza; (3,4) id 4 fora; (4,5) ambos fora -> 1 cruzamento
    # fold1: par (0,3) dentro treino; (3,4) cruza; (4,5) dentro teste
    assert resultado["agregado"]["pares_cruzando_treino_teste"] == 2
