"""Testes para `src.eda.duplicatas`."""
from __future__ import annotations

import pandas as pd

from src.eda.duplicatas import (
    componentes_conexos,
    duplicatas_exatas,
    encontrar_near_duplicates,
    notificar_afetadas_exatas,
    notificar_afetadas_near,
    resumo_duplicatas_exatas,
)


def _fixture_com_duplicata_exata_e_near() -> pd.DataFrame:
    """6 linhas: duas com mesmo `link` exato; duas com texto quase idêntico;
    duas originais."""
    return pd.DataFrame(
        [
            {"link": "https://a/1", "title": "Primeira notícia", "text": "Texto original completamente único sobre uma política nova apresentada hoje pelo governo."},
            {"link": "https://a/1", "title": "Primeira notícia", "text": "Texto original completamente único sobre uma política nova apresentada hoje pelo governo."},
            {"link": "https://a/3", "title": "Quase duplicata A", "text": "A economia brasileira cresceu três por cento no trimestre passado segundo dados do instituto."},
            {"link": "https://a/4", "title": "Quase duplicata B", "text": "A economia brasileira cresceu tres por cento no trimestre passado segundo dados do instituto."},
            {"link": "https://a/5", "title": "Tema totalmente outro", "text": "Um meteorito passou perto da terra causando fenômeno óptico raro no céu noturno de hoje."},
            {"link": "https://a/6", "title": "Diferente também", "text": "O festival de cinema ofereceu sessões gratuitas durante todo o fim de semana para o público."},
        ]
    )


def test_duplicatas_exatas_conta_grupos():
    df = _fixture_com_duplicata_exata_e_near()
    relatorio = duplicatas_exatas(df, "link")
    # 2 linhas com 'https://a/1', então 1 grupo duplicado, 1 linha adicional
    assert relatorio["grupos"] == 1
    assert relatorio["linhas_duplicadas"] == 1
    assert len(relatorio["exemplos"]) == 1


def test_duplicatas_exatas_titulo_sem_duplicatas():
    df = _fixture_com_duplicata_exata_e_near()
    # Títulos: linhas 0 e 1 compartilham "Primeira notícia"; os demais
    # são únicos.
    relatorio = duplicatas_exatas(df, "title")
    assert relatorio["linhas_duplicadas"] == 1
    assert relatorio["grupos"] == 1


def test_resumo_duplicatas_exatas_formato():
    df = _fixture_com_duplicata_exata_e_near()
    resumo = resumo_duplicatas_exatas(df, ["link", "title", "text"])
    assert set(resumo.columns) == {"tipo", "linhas_duplicadas", "grupos", "pct_corpus", "exemplos_json"}
    # A linha 'link' deve ter 1 duplicada; pct_corpus = 1/6
    linha_link = resumo.loc[resumo["tipo"] == "link"].iloc[0]
    assert linha_link["linhas_duplicadas"] == 1
    assert linha_link["pct_corpus"] == 1 / 6


def test_encontrar_near_duplicates_detecta_par_plantado():
    df = _fixture_com_duplicata_exata_e_near()
    pares = encontrar_near_duplicates(df, coluna_texto="text", threshold=0.7)
    # Esperamos pelo menos o par (0, 1) — exatas são 100% near-duplicatas;
    # e/ou (2, 3) — "três" vs "tres" (diferença mínima).
    pares_conjunto = set(zip(pares["id_a"].tolist(), pares["id_b"].tolist()))
    # exata: (0, 1)
    assert (0, 1) in pares_conjunto
    # near (acento): (2, 3)
    assert (2, 3) in pares_conjunto


def test_encontrar_near_duplicates_threshold_alto_remove_par_fraco():
    """Threshold muito alto deve descartar pares que não são quase idênticos."""
    df = pd.DataFrame(
        [
            {"text": "O time jogou bem na partida de ontem e conseguiu uma vitória importante no campeonato."},
            {"text": "A equipe jogou mal na partida de ontem e perdeu de goleada no torneio regional."},
        ]
    )
    # Textos com algumas palavras em comum mas narrativas diferentes.
    pares = encontrar_near_duplicates(df, coluna_texto="text", threshold=0.85)
    assert pares.empty


def test_componentes_conexos_agrupa():
    pares = pd.DataFrame(
        [
            {"id_a": 1, "id_b": 2},
            {"id_a": 2, "id_b": 3},
            {"id_a": 5, "id_b": 6},
        ]
    )
    componentes = componentes_conexos(pares)
    componentes_sorted = sorted(componentes, key=lambda g: g[0])
    assert componentes_sorted == [[1, 2, 3], [5, 6]]


def test_notificar_afetadas_near_mantem_canonica():
    pares = pd.DataFrame(
        [
            {"id_a": 1, "id_b": 2},
            {"id_a": 2, "id_b": 3},
        ]
    )
    afetadas = notificar_afetadas_near(pares)
    # Componente {1, 2, 3}: canônica = 1; afetadas = {2, 3}
    assert afetadas == {2, 3}


def test_notificar_afetadas_exatas_conta_repetidos():
    df = pd.DataFrame(
        [
            {"link": "a", "title": "x"},
            {"link": "a", "title": "y"},  # duplicada por link
            {"link": "b", "title": "z"},
            {"link": "c", "title": "x"},  # duplicada por title com linha 0
        ]
    )
    afetadas = notificar_afetadas_exatas(df, ["link", "title"])
    # Índice 1 duplicado por link, índice 3 por title (mantém 0 como canônico).
    assert afetadas == {1, 3}
