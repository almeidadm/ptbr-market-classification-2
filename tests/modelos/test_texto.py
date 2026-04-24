"""Testes para `src.modelos.texto`."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.modelos.texto import juntar_titulo_texto, preparar_entrada


def test_concatena_com_espaco_unico() -> None:
    assert juntar_titulo_texto("Título", "Corpo") == "Título Corpo"


def test_strip_em_ambas_as_partes() -> None:
    assert juntar_titulo_texto("  Título  ", "  Corpo  ") == "Título Corpo"


def test_nulo_em_title_usa_somente_text() -> None:
    assert juntar_titulo_texto(np.nan, "Corpo") == "Corpo"


def test_nulo_em_text_usa_somente_title() -> None:
    assert juntar_titulo_texto("Título", np.nan) == "Título"


def test_ambos_nulos_retorna_string_vazia() -> None:
    assert juntar_titulo_texto(np.nan, np.nan) == ""


def test_string_vazia_ou_espacos_e_tratada_como_nulo_logico() -> None:
    assert juntar_titulo_texto("", "Corpo") == "Corpo"
    assert juntar_titulo_texto("  ", "Corpo") == "Corpo"


def test_preparar_entrada_aplica_sobre_df() -> None:
    df = pd.DataFrame(
        {
            "title": ["A", None, "C"],
            "text": ["x", "y", None],
        }
    )
    serie = preparar_entrada(df)
    assert serie.tolist() == ["A x", "y", "C"]
