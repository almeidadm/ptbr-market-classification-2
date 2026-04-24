"""Testes para `src.preprocessamento.recortes` (ADR 004)."""
from __future__ import annotations

import pandas as pd
import pytest

from src.config import CLASSES_OPCAO_3, CLASSES_OPCAO_7
from src.preprocessamento.recortes import (
    aplicar_opcao_3,
    aplicar_opcao_4,
    aplicar_opcao_7,
)


def _corpus_sintetico() -> pd.DataFrame:
    # Corpus com várias classes em proporções controladas para testar
    # o pipeline da Opção 7 (threshold ≥1.000 + relabel).
    #   - 7 âncoras da Opção 7 com 1.100 amostras cada (acima do threshold).
    #   - 2 classes >= threshold mas fora das âncoras (vão para `outros`).
    #   - 2 classes < threshold (removidas do corpus).
    linhas = []
    for classe in CLASSES_OPCAO_7:
        if classe == "outros":
            continue
        for i in range(1100):
            linhas.append({"y_original": classe, "link": f"{classe}-{i}"})
    for i in range(1200):
        linhas.append({"y_original": "opiniao", "link": f"opiniao-{i}"})
    for i in range(1050):
        linhas.append({"y_original": "tec", "link": f"tec-{i}"})
    for i in range(50):
        linhas.append({"y_original": "musica", "link": f"musica-{i}"})
    for i in range(30):
        linhas.append({"y_original": "bichos", "link": f"bichos-{i}"})
    return pd.DataFrame(linhas)


def test_opcao_7_colapsa_classes_nao_ancoras_em_outros() -> None:
    df = _corpus_sintetico()
    resultado = aplicar_opcao_7(df, threshold=1000)

    assert set(resultado["y_colapsado"].unique()) <= set(CLASSES_OPCAO_7)
    assert "outros" in resultado["y_colapsado"].unique()
    # Só `opiniao` e `tec` devem sobreviver para virar `outros`
    # (`musica` e `bichos` são filtradas pelo threshold).
    assert (resultado["y_colapsado"] == "outros").sum() == 1200 + 1050


def test_opcao_7_preserva_y_original_nas_sobreviventes() -> None:
    df = _corpus_sintetico()
    resultado = aplicar_opcao_7(df, threshold=1000)
    assert "opiniao" in resultado["y_original"].unique()
    assert "tec" in resultado["y_original"].unique()


def test_opcao_7_remove_linhas_abaixo_do_threshold() -> None:
    df = _corpus_sintetico()
    resultado = aplicar_opcao_7(df, threshold=1000)
    # Removeu 50 `musica` + 30 `bichos` = 80 linhas.
    assert len(resultado) == len(df) - 50 - 30
    assert "musica" not in resultado["y_original"].unique()
    assert "bichos" not in resultado["y_original"].unique()


def test_opcao_4_exclui_colunas_e_ilustrada_mesmo_acima_do_threshold() -> None:
    df = pd.DataFrame(
        {
            "y_original": (["colunas"] * 600 + ["ilustrada"] * 700 + ["poder"] * 800 + ["mercado"] * 500),
            "link": [f"l{i}" for i in range(600 + 700 + 800 + 500)],
        }
    )
    resultado = aplicar_opcao_4(df, threshold=500)
    assert set(resultado["y_colapsado"].unique()) == {"poder", "mercado"}
    assert len(resultado) == 800 + 500


def test_opcao_4_filtra_abaixo_do_threshold() -> None:
    df = pd.DataFrame(
        {
            "y_original": (["poder"] * 600 + ["tec"] * 400 + ["mercado"] * 500),
            "link": [f"l{i}" for i in range(600 + 400 + 500)],
        }
    )
    resultado = aplicar_opcao_4(df, threshold=500)
    assert "tec" not in resultado["y_colapsado"].unique()
    assert len(resultado) == 1100


def test_opcao_3_mantem_somente_as_5_classes_canonicas() -> None:
    df = pd.DataFrame(
        {
            "y_original": (
                ["poder"] * 10 + ["mercado"] * 20 + ["esporte"] * 15
                + ["mundo"] * 5 + ["cotidiano"] * 30 + ["colunas"] * 100
                + ["tec"] * 50
            ),
            "link": [f"l{i}" for i in range(230)],
        }
    )
    resultado = aplicar_opcao_3(df)
    assert set(resultado["y_colapsado"].unique()) == set(CLASSES_OPCAO_3)
    assert len(resultado) == 10 + 20 + 15 + 5 + 30


def test_y_colapsado_igual_a_y_original_em_opcao_4_e_3() -> None:
    df = pd.DataFrame(
        {
            "y_original": ["poder"] * 600 + ["mercado"] * 600,
            "link": [f"l{i}" for i in range(1200)],
        }
    )
    r4 = aplicar_opcao_4(df, threshold=500)
    r3 = aplicar_opcao_3(df)
    assert (r4["y_colapsado"] == r4["y_original"]).all()
    assert (r3["y_colapsado"] == r3["y_original"]).all()
