"""
Contagens e prevalências temporais para a EDA.

`prevalencia_por_periodo` agrega por `pd.Grouper(freq=...)` a uma das
granularidades `'D'` (diária), `'W'` (semanal) ou `'M'` (mensal, final de
mês). A granularidade principal (mensal) é a definida em 002; diária e
semanal ficam para o apêndice.
"""
from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from scipy import stats

Granularidade = Literal["D", "W", "M"]
GRANULARIDADES_VALIDAS: tuple[Granularidade, ...] = ("D", "W", "M")


def _validar_granularidade(granularidade: str) -> None:
    if granularidade not in GRANULARIDADES_VALIDAS:
        raise ValueError(
            f"granularidade deve ser uma de {GRANULARIDADES_VALIDAS}, "
            f"recebido: {granularidade!r}"
        )


def serie_total_e_classe(
    df: pd.DataFrame,
    coluna_alvo_bool: str,
    granularidade: Granularidade,
    coluna_data: str = "date",
) -> pd.DataFrame:
    """Retorna DataFrame com índice temporal e colunas `total`, `positivos`, `prevalencia`.

    Linhas com data nula são ignoradas. `prevalencia` é `positivos / total`;
    períodos com `total == 0` recebem `prevalencia` nula.
    """
    _validar_granularidade(granularidade)

    if coluna_alvo_bool not in df.columns:
        raise KeyError(f"coluna alvo não encontrada: {coluna_alvo_bool}")
    if coluna_data not in df.columns:
        raise KeyError(f"coluna de data não encontrada: {coluna_data}")

    sub = df.loc[df[coluna_data].notna(), [coluna_data, coluna_alvo_bool]].copy()
    sub[coluna_alvo_bool] = sub[coluna_alvo_bool].astype(bool)

    # `'M'` foi deprecado em pandas 2.2 em favor de `'ME'`; a API pública
    # do módulo mantém `'M'` como aliás por retrocompatibilidade com o plano 002.
    freq_pandas = {"D": "D", "W": "W", "M": "ME"}[granularidade]
    grupos = sub.groupby(
        pd.Grouper(key=coluna_data, freq=freq_pandas),
        sort=True,
    )
    total = grupos.size().rename("total")
    positivos = grupos[coluna_alvo_bool].sum().astype("int64").rename("positivos")

    resultado = pd.concat([total, positivos], axis=1)
    resultado["prevalencia"] = resultado["positivos"].div(resultado["total"])
    return resultado


def prevalencia_por_periodo(
    df: pd.DataFrame,
    alvo: str,
    granularidade: Granularidade,
    coluna_data: str = "date",
) -> pd.DataFrame:
    """Alias semântico para `serie_total_e_classe` orientado à prevalência.

    Mantido como função separada para o notebook chamar o nome que faz
    sentido no texto do artigo (prevalência) quando o interesse é a taxa,
    e `serie_total_e_classe` quando o interesse inclui volume bruto.
    """
    return serie_total_e_classe(df, alvo, granularidade, coluna_data)


def intervalo_wilson(
    positivos: int,
    total: int,
    confianca: float = 0.95,
) -> tuple[float, float, float]:
    """Intervalo de confiança de Wilson para proporção binomial.

    Retorna `(prevalencia, limite_inferior, limite_superior)`. Wilson é
    preferido ao IC normal (Wald) quando a proporção é pequena — caso
    deste projeto, em que a prevalência esperada é ~12,6%.
    """
    if total <= 0:
        return (float("nan"), float("nan"), float("nan"))
    if not 0 < confianca < 1:
        raise ValueError("confianca deve estar em (0, 1)")

    p = positivos / total
    z = float(stats.norm.ppf(0.5 + confianca / 2.0))
    denominador = 1.0 + (z**2) / total
    centro = (p + (z**2) / (2.0 * total)) / denominador
    margem = (z * float(np.sqrt(p * (1.0 - p) / total + (z**2) / (4.0 * total**2)))) / denominador
    return (float(p), float(centro - margem), float(centro + margem))


# Convenção: segunda = 0 ... domingo = 6, consistente com `pd.Timestamp.weekday`.
_NOMES_DIAS_SEMANA_PT = ["seg", "ter", "qua", "qui", "sex", "sáb", "dom"]


def prevalencia_por_dia_semana(
    df: pd.DataFrame,
    coluna_alvo_bool: str,
    coluna_data: str = "date",
) -> pd.DataFrame:
    """Prevalência de `coluna_alvo_bool` agregada por dia-da-semana.

    Retorna DataFrame ordenado seg→dom, com colunas `dia_semana` (0..6),
    `nome` (rótulo PT-BR curto), `total`, `positivos`, `prevalencia`,
    `ic_inf`, `ic_sup` (IC 95% Wilson por dia). Linhas com data nula são
    descartadas. Dias-da-semana sem observação não aparecem — no corpus
    esperado (2015–2017) todos os 7 dias comparecem.
    """
    if coluna_alvo_bool not in df.columns:
        raise KeyError(f"coluna alvo não encontrada: {coluna_alvo_bool}")
    if coluna_data not in df.columns:
        raise KeyError(f"coluna de data não encontrada: {coluna_data}")

    sub = df.loc[df[coluna_data].notna(), [coluna_data, coluna_alvo_bool]].copy()
    sub[coluna_alvo_bool] = sub[coluna_alvo_bool].astype(bool)
    sub["dia_semana"] = sub[coluna_data].dt.weekday.astype("int64")

    grupos = sub.groupby("dia_semana", sort=True)
    total = grupos.size().rename("total").astype("int64")
    positivos = grupos[coluna_alvo_bool].sum().astype("int64").rename("positivos")

    frame = pd.concat([total, positivos], axis=1).reset_index()
    frame["nome"] = frame["dia_semana"].map(lambda d: _NOMES_DIAS_SEMANA_PT[int(d)])
    frame["prevalencia"] = frame["positivos"] / frame["total"]

    ic_inf = []
    ic_sup = []
    for _, linha in frame.iterrows():
        _, baixo, alto = intervalo_wilson(int(linha["positivos"]), int(linha["total"]))
        ic_inf.append(baixo)
        ic_sup.append(alto)
    frame["ic_inf"] = ic_inf
    frame["ic_sup"] = ic_sup

    return frame[["dia_semana", "nome", "total", "positivos", "prevalencia", "ic_inf", "ic_sup"]]


def prevalencia_por_mes_ano(
    df: pd.DataFrame,
    coluna_alvo_bool: str,
    coluna_data: str = "date",
) -> pd.DataFrame:
    """Prevalência agregada por célula (ano, mês). Formato longo.

    Retorna DataFrame com uma linha por célula (ano, mês) presente no
    corpus, colunas `ano`, `mes`, `total`, `positivos`, `prevalencia`,
    ordenado por `(ano, mes)`. Células ausentes (meses completamente
    fora do corpus) não aparecem — a figura é responsável por decidir
    como apresentar lacunas.
    """
    if coluna_alvo_bool not in df.columns:
        raise KeyError(f"coluna alvo não encontrada: {coluna_alvo_bool}")
    if coluna_data not in df.columns:
        raise KeyError(f"coluna de data não encontrada: {coluna_data}")

    sub = df.loc[df[coluna_data].notna(), [coluna_data, coluna_alvo_bool]].copy()
    sub[coluna_alvo_bool] = sub[coluna_alvo_bool].astype(bool)
    sub["ano"] = sub[coluna_data].dt.year.astype("int64")
    sub["mes"] = sub[coluna_data].dt.month.astype("int64")

    grupos = sub.groupby(["ano", "mes"], sort=True)
    total = grupos.size().rename("total").astype("int64")
    positivos = grupos[coluna_alvo_bool].sum().astype("int64").rename("positivos")

    frame = pd.concat([total, positivos], axis=1).reset_index()
    frame["prevalencia"] = frame["positivos"] / frame["total"]
    return frame[["ano", "mes", "total", "positivos", "prevalencia"]]
