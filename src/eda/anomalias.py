"""
Detecção de lacunas e anomalias na série diária de notícias.

Este módulo trabalha sobre a série de **contagem total de notícias por
dia** (não sobre prevalência). A ideia é caracterizar o padrão editorial
do corpus: quando é que o veículo publica, e quando há picos/quedas
anômalas em relação ao seu próprio comportamento típico.

Duas regras de anomalia são reportadas em paralelo, em colunas
separadas, deixando para ciclo futuro a decisão de qual entra no artigo:

- **z-score clássico** sobre dias com publicação (ignora dias-zero). Usa
  média e desvio-padrão amostral (Bessel, `ddof=1`).
- **z-score robusto via MAD**: `z_rob = 0.6745 * (x - mediana) / MAD`,
  onde `MAD = mediana(|x - mediana(x)|)`. A constante `0.6745` é o
  quantil 0,75 da normal padrão e faz com que, sob normalidade, `z_rob`
  tenha variância aproximadamente unitária — comparável ao z clássico.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Constante 0.6745 ≈ Phi^{-1}(0.75): sob N(0,1), MAD = 0.6745 → escala
# equivalente ao desvio-padrão. Mantida como constante nomeada para que
# o leitor não precise reconsultar literatura ao revisar o código.
_FATOR_MAD_NORMAL = 0.6745


def gerar_calendario(
    df: pd.DataFrame,
    coluna_data: str = "date",
) -> pd.DataFrame:
    """Retorna a série diária reindexada ao calendário completo.

    Conta o número de linhas por dia e preenche dias ausentes com zero.
    Linhas com data nula são descartadas antes da agregação. Retorna
    DataFrame com colunas `data` (datetime normalizada) e `total` (int).
    """
    if coluna_data not in df.columns:
        raise KeyError(f"coluna de data não encontrada: {coluna_data}")

    datas = df.loc[df[coluna_data].notna(), coluna_data].dt.normalize()
    if datas.empty:
        return pd.DataFrame({"data": pd.to_datetime([]), "total": pd.Series([], dtype="int64")})

    contagem = datas.value_counts().sort_index(kind="mergesort")
    indice_completo = pd.date_range(
        start=contagem.index.min(),
        end=contagem.index.max(),
        freq="D",
    )
    reindexada = contagem.reindex(indice_completo, fill_value=0).astype("int64")
    return pd.DataFrame({"data": reindexada.index, "total": reindexada.values})


def detectar_dias_zero(df_calendario: pd.DataFrame) -> pd.DataFrame:
    """Lista os dias do calendário sem publicação alguma (`total == 0`).

    Retorna DataFrame com colunas `data` e `dia_semana` (nome curto
    PT-BR), ordenado por data. Calendário vazio retorna frame vazio.
    """
    if df_calendario.empty:
        return pd.DataFrame({"data": pd.to_datetime([]), "dia_semana": []})

    zeros = df_calendario.loc[df_calendario["total"] == 0, ["data"]].copy()
    zeros["dia_semana"] = zeros["data"].dt.day_name(locale=None)
    # `day_name` depende de locale; para saída estável em PT-BR usamos
    # a abreviação curta do mapeamento já usado em `contagens`.
    nomes_pt = ["seg", "ter", "qua", "qui", "sex", "sáb", "dom"]
    zeros["dia_semana"] = zeros["data"].dt.weekday.map(lambda d: nomes_pt[int(d)])
    return zeros.reset_index(drop=True)


def detectar_anomalias_diarias(
    df_calendario: pd.DataFrame,
    limiar: float = 3.0,
) -> pd.DataFrame:
    """Marca dias anômalos pelo z clássico E pelo z robusto (MAD).

    Estatísticas calculadas **apenas sobre dias com publicação**
    (`total > 0`): a distribuição alvo é "dado que publicou, quanto
    publicou". Dias-zero ficam em `detectar_dias_zero` e não entram no
    cálculo nem aparecem no retorno.

    Devolve DataFrame com um subconjunto do calendário (apenas dias
    flagados por pelo menos uma das duas regras), com colunas:
    `data`, `total`, `z_classico`, `z_robusto`, `flag_classico`,
    `flag_robusto`, `flag_ambos`. Ordenação por `data` crescente.

    `limiar` é aplicado ao valor absoluto: anomalias acima e abaixo.
    """
    if df_calendario.empty:
        colunas = [
            "data",
            "total",
            "z_classico",
            "z_robusto",
            "flag_classico",
            "flag_robusto",
            "flag_ambos",
        ]
        return pd.DataFrame({c: [] for c in colunas})

    frame = df_calendario.copy()
    frame["z_classico"] = np.nan
    frame["z_robusto"] = np.nan

    mascara_publicou = frame["total"] > 0
    valores = frame.loc[mascara_publicou, "total"].to_numpy(dtype=float)

    if valores.size >= 2:
        media = float(np.mean(valores))
        desvio = float(np.std(valores, ddof=1))
        if desvio > 0.0:
            frame.loc[mascara_publicou, "z_classico"] = (valores - media) / desvio

    if valores.size >= 1:
        mediana = float(np.median(valores))
        mad = float(np.median(np.abs(valores - mediana)))
        if mad > 0.0:
            frame.loc[mascara_publicou, "z_robusto"] = (
                _FATOR_MAD_NORMAL * (valores - mediana) / mad
            )

    frame["flag_classico"] = frame["z_classico"].abs() > limiar
    frame["flag_robusto"] = frame["z_robusto"].abs() > limiar
    frame["flag_classico"] = frame["flag_classico"].fillna(False)
    frame["flag_robusto"] = frame["flag_robusto"].fillna(False)
    frame["flag_ambos"] = frame["flag_classico"] & frame["flag_robusto"]

    anomalias = frame.loc[frame["flag_classico"] | frame["flag_robusto"]].copy()
    anomalias = anomalias.sort_values(by="data", kind="mergesort").reset_index(drop=True)
    return anomalias[
        [
            "data",
            "total",
            "z_classico",
            "z_robusto",
            "flag_classico",
            "flag_robusto",
            "flag_ambos",
        ]
    ]
