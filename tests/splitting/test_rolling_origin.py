"""Testes para `src.splitting.rolling_origin` (ADR 008)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.splitting.rolling_origin import gerar_folds_rolling


def _corpus_mensal(data_inicio: str, n_meses: int, por_mes: int = 30) -> pd.DataFrame:
    linhas = []
    inicio = pd.Timestamp(data_inicio)
    for m in range(n_meses):
        dia_referencia = (inicio + pd.DateOffset(months=m)).replace(day=1)
        for d in range(por_mes):
            linhas.append(
                {
                    "date": dia_referencia + pd.Timedelta(days=d),
                    "link": f"m{m:02d}-d{d:02d}",
                    "y_colapsado": "mercado" if d % 3 == 0 else "outros",
                }
            )
    return pd.DataFrame(linhas)


def test_cinco_folds_com_janelas_de_tres_meses_nao_sobrepostas() -> None:
    df = _corpus_mensal("2015-01-01", n_meses=33)
    folds = gerar_folds_rolling(df)
    assert len(folds) == 5

    inicios_esperados = [
        pd.Timestamp("2016-07-01"),
        pd.Timestamp("2016-10-01"),
        pd.Timestamp("2017-01-01"),
        pd.Timestamp("2017-04-01"),
        pd.Timestamp("2017-07-01"),
    ]
    assert [f.inicio_teste for f in folds] == inicios_esperados


def test_treino_sempre_antes_do_teste_em_cada_fold() -> None:
    df = _corpus_mensal("2015-01-01", n_meses=33)
    folds = gerar_folds_rolling(df)
    for fold in folds:
        datas_treino = pd.to_datetime(df.iloc[fold.treino_idx]["date"])
        datas_teste = pd.to_datetime(df.iloc[fold.teste_idx]["date"])
        assert datas_treino.max() < datas_teste.min()


def test_inner_val_cobre_ultimos_tres_meses_do_treino() -> None:
    df = _corpus_mensal("2015-01-01", n_meses=33)
    folds = gerar_folds_rolling(df)
    for fold in folds:
        datas_inner_train = pd.to_datetime(df.iloc[fold.inner_train_idx]["date"])
        datas_inner_val = pd.to_datetime(df.iloc[fold.inner_val_idx]["date"])
        assert datas_inner_train.max() < datas_inner_val.min()
        assert (datas_inner_val.max() < fold.inicio_teste)


def test_expanding_window_treino_cresce_a_cada_fold() -> None:
    df = _corpus_mensal("2015-01-01", n_meses=33)
    folds = gerar_folds_rolling(df)
    tamanhos = [len(f.treino_idx) for f in folds]
    assert tamanhos == sorted(tamanhos)
    assert all(a < b for a, b in zip(tamanhos, tamanhos[1:]))


def test_determinismo_sob_duas_execucoes() -> None:
    df = _corpus_mensal("2015-01-01", n_meses=33)
    f1 = gerar_folds_rolling(df)
    f2 = gerar_folds_rolling(df)
    for a, b in zip(f1, f2):
        np.testing.assert_array_equal(a.treino_idx, b.treino_idx)
        np.testing.assert_array_equal(a.teste_idx, b.teste_idx)
        np.testing.assert_array_equal(a.inner_train_idx, b.inner_train_idx)
        np.testing.assert_array_equal(a.inner_val_idx, b.inner_val_idx)


def test_corpus_curto_demais_erra_claro() -> None:
    df = _corpus_mensal("2015-01-01", n_meses=12)
    with pytest.raises(ValueError, match="teste vazio"):
        gerar_folds_rolling(df)


def test_treino_e_teste_disjuntos_em_todos_os_folds() -> None:
    df = _corpus_mensal("2015-01-01", n_meses=33)
    folds = gerar_folds_rolling(df)
    for fold in folds:
        intersecao = np.intersect1d(fold.treino_idx, fold.teste_idx)
        assert len(intersecao) == 0


def test_inner_train_e_inner_val_cobrem_exatamente_o_treino() -> None:
    df = _corpus_mensal("2015-01-01", n_meses=33)
    folds = gerar_folds_rolling(df)
    for fold in folds:
        juntos = np.sort(np.concatenate([fold.inner_train_idx, fold.inner_val_idx]))
        np.testing.assert_array_equal(juntos, np.sort(fold.treino_idx))
