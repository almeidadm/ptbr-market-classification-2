"""
Funções puras de figuras para a EDA.

Cada função devolve um `matplotlib.figure.Figure`. Não chamam
`plt.show()` nem salvam em disco — a persistência e exibição ficam no
notebook ou script chamador.
"""
from __future__ import annotations

import matplotlib.dates as mdates
import matplotlib.figure as mfigure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.eda.contagens import intervalo_wilson


def figura_distribuicao_categorias(
    value_counts: pd.Series,
    top_n: int = 7,
) -> mfigure.Figure:
    """Barras horizontais da distribuição de categorias com política top-N + "outras".

    `value_counts` deve ser a saída de `df['category'].value_counts(dropna=False)`
    (ou similar) — índice com categorias, valores com contagens, já
    ordenada de forma decrescente. Categorias fora do top-N entram
    agregadas em "outras".
    """
    if value_counts.empty:
        raise ValueError("value_counts está vazio")

    top = value_counts.head(top_n)
    restante = value_counts.iloc[top_n:].sum()
    if restante > 0:
        dados = pd.concat([top, pd.Series({"outras": int(restante)})])
    else:
        dados = top

    dados = dados.iloc[::-1]

    figura, eixo = plt.subplots(figsize=(8, max(3, 0.4 * len(dados) + 1)))
    eixo.barh(dados.index.astype(str), dados.values, color="#4C72B0")
    eixo.set_xlabel("Número de notícias")
    eixo.set_ylabel("Categoria")
    eixo.set_title(f"Distribuição de categorias (top-{top_n} + outras)")
    total = int(value_counts.sum())
    for indice, valor in enumerate(dados.values):
        eixo.text(
            valor,
            indice,
            f" {valor:,} ({valor / total:.1%})".replace(",", "."),
            va="center",
            fontsize=9,
        )
    eixo.margins(x=0.15)
    figura.tight_layout()
    return figura


def _formatar_eixo_tempo_mensal(eixo) -> None:
    eixo.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    eixo.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    for rotulo in eixo.get_xticklabels():
        rotulo.set_rotation(45)
        rotulo.set_ha("right")


def figura_serie_mensal_prevalencia(
    serie: pd.DataFrame,
    titulo: str = "Prevalência mensal de 'mercado' (IC 95% Wilson por ponto)",
) -> mfigure.Figure:
    """Linha de prevalência mensal com banda IC 95% Wilson por ponto.

    `serie` deve ter índice temporal (ou coluna `data`) e colunas
    `total`, `positivos`, `prevalencia`. A banda é calculada ponto a
    ponto com `intervalo_wilson`; períodos com `total == 0` ficam como
    gaps na linha e não desenham banda.
    """
    dados = _coerce_serie_para_indice_temporal(serie)

    datas = dados.index.to_pydatetime()
    prevalencia = dados["prevalencia"].to_numpy(dtype=float)
    totais = dados["total"].to_numpy(dtype=float)
    positivos = dados["positivos"].to_numpy(dtype=float)

    inferior = np.full_like(prevalencia, np.nan)
    superior = np.full_like(prevalencia, np.nan)
    for idx in range(len(dados)):
        if totais[idx] > 0:
            _, baixo, alto = intervalo_wilson(int(positivos[idx]), int(totais[idx]))
            inferior[idx] = baixo
            superior[idx] = alto

    figura, eixo = plt.subplots(figsize=(10, 4.5))
    eixo.fill_between(
        datas,
        inferior,
        superior,
        color="#4C72B0",
        alpha=0.18,
        label="IC 95% Wilson",
    )
    eixo.plot(
        datas,
        prevalencia,
        color="#4C72B0",
        marker="o",
        linewidth=1.5,
        markersize=4,
        label="Prevalência observada",
    )
    eixo.set_ylabel("Prevalência de 'mercado'")
    eixo.set_xlabel("Mês")
    eixo.set_title(titulo)
    eixo.set_ylim(bottom=0.0)
    eixo.grid(True, axis="y", alpha=0.3)
    eixo.legend(loc="best", frameon=False)
    _formatar_eixo_tempo_mensal(eixo)
    figura.tight_layout()
    return figura


def figura_serie_mensal_contagens(
    serie: pd.DataFrame,
    titulo_total: str = "Total de notícias por mês",
    titulo_positivos: str = "Notícias de 'mercado' por mês",
) -> mfigure.Figure:
    """Dois subplots empilhados compartilhando eixo X: total e positivos por mês."""
    dados = _coerce_serie_para_indice_temporal(serie)
    datas = dados.index.to_pydatetime()

    figura, (eixo_total, eixo_positivos) = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(10, 6),
        sharex=True,
    )
    eixo_total.plot(
        datas,
        dados["total"].to_numpy(),
        color="#4C72B0",
        marker="o",
        linewidth=1.5,
        markersize=4,
    )
    eixo_total.set_ylabel("Total")
    eixo_total.set_title(titulo_total)
    eixo_total.grid(True, axis="y", alpha=0.3)

    eixo_positivos.plot(
        datas,
        dados["positivos"].to_numpy(),
        color="#C44E52",
        marker="o",
        linewidth=1.5,
        markersize=4,
    )
    eixo_positivos.set_ylabel("Positivos")
    eixo_positivos.set_xlabel("Mês")
    eixo_positivos.set_title(titulo_positivos)
    eixo_positivos.grid(True, axis="y", alpha=0.3)

    _formatar_eixo_tempo_mensal(eixo_positivos)
    figura.tight_layout()
    return figura


def figura_serie_semanal_prevalencia(
    serie: pd.DataFrame,
    titulo: str = "Prevalência semanal de 'mercado' (apêndice)",
) -> mfigure.Figure:
    """Linha de prevalência semanal. Sem banda IC (série mais longa, mais ruidosa)."""
    dados = _coerce_serie_para_indice_temporal(serie)

    figura, eixo = plt.subplots(figsize=(10, 4))
    eixo.plot(
        dados.index.to_pydatetime(),
        dados["prevalencia"].to_numpy(),
        color="#4C72B0",
        linewidth=1.0,
    )
    eixo.set_ylabel("Prevalência de 'mercado'")
    eixo.set_xlabel("Semana")
    eixo.set_title(titulo)
    eixo.set_ylim(bottom=0.0)
    eixo.grid(True, axis="y", alpha=0.3)
    eixo.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    eixo.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    for rotulo in eixo.get_xticklabels():
        rotulo.set_rotation(45)
        rotulo.set_ha("right")
    figura.tight_layout()
    return figura


def figura_serie_diaria_prevalencia(
    serie: pd.DataFrame,
    titulo: str = "Prevalência diária de 'mercado' (apêndice, ruidoso)",
) -> mfigure.Figure:
    """Linha de prevalência diária. Espera-se ruído alto — figura de referência."""
    dados = _coerce_serie_para_indice_temporal(serie)

    figura, eixo = plt.subplots(figsize=(10, 4))
    eixo.plot(
        dados.index.to_pydatetime(),
        dados["prevalencia"].to_numpy(),
        color="#4C72B0",
        linewidth=0.5,
        alpha=0.8,
    )
    eixo.set_ylabel("Prevalência de 'mercado'")
    eixo.set_xlabel("Dia")
    eixo.set_title(titulo)
    eixo.set_ylim(bottom=0.0)
    eixo.grid(True, axis="y", alpha=0.3)
    eixo.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    eixo.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    for rotulo in eixo.get_xticklabels():
        rotulo.set_rotation(45)
        rotulo.set_ha("right")
    figura.tight_layout()
    return figura


def figura_prevalencia_dia_semana(
    df_dia_semana: pd.DataFrame,
    prevalencia_global: float = 0.1255,
    titulo: str = "Prevalência de 'mercado' por dia-da-semana (IC 95% Wilson)",
) -> mfigure.Figure:
    """Barras da prevalência por dia-da-semana com erro IC 95% Wilson.

    Espera `df_dia_semana` no formato devolvido por
    `prevalencia_por_dia_semana`: colunas `dia_semana, nome, total,
    positivos, prevalencia, ic_inf, ic_sup`. Desenha uma linha tracejada
    horizontal em `prevalencia_global` como referência.
    """
    dados = df_dia_semana.sort_values(by="dia_semana", kind="mergesort").reset_index(drop=True)

    prevalencia = dados["prevalencia"].to_numpy(dtype=float)
    inferior = dados["ic_inf"].to_numpy(dtype=float)
    superior = dados["ic_sup"].to_numpy(dtype=float)
    # `yerr` para `bar` aceita distâncias absolutas do centro — não os
    # limites. Convertendo aqui.
    erro_baixo = prevalencia - inferior
    erro_alto = superior - prevalencia
    rotulos = dados["nome"].tolist()

    figura, eixo = plt.subplots(figsize=(7, 4))
    eixo.bar(
        rotulos,
        prevalencia,
        yerr=[erro_baixo, erro_alto],
        color="#4C72B0",
        capsize=4,
        ecolor="#2f3e50",
    )
    eixo.axhline(
        prevalencia_global,
        color="#C44E52",
        linestyle="--",
        linewidth=1.2,
        label=f"Prevalência global ({prevalencia_global:.2%})",
    )
    eixo.set_xlabel("Dia da semana")
    eixo.set_ylabel("Prevalência de 'mercado'")
    eixo.set_title(titulo)
    eixo.set_ylim(bottom=0.0)
    eixo.grid(True, axis="y", alpha=0.3)
    eixo.legend(loc="best", frameon=False)
    figura.tight_layout()
    return figura


def figura_heatmap_prevalencia_mes_ano(
    df_mes_ano: pd.DataFrame,
    n_minimo: int = 100,
    titulo: str = "Prevalência de 'mercado' por mês e ano",
) -> mfigure.Figure:
    """Heatmap ano (coluna) × mês (linha) da prevalência de `mercado`.

    Células com `total < n_minimo` são pintadas em cinza (indicando que
    o mês é parcial e a taxa observada é pouco informativa); não entram
    na escala de cor. Anotação numérica só é desenhada quando o número
    de células for pequeno (≤ 40) — caso contrário o heatmap fica
    poluído e o colormap já carrega a informação.
    """
    if df_mes_ano.empty:
        raise ValueError("df_mes_ano está vazio")

    matriz_prev = df_mes_ano.pivot(index="mes", columns="ano", values="prevalencia")
    matriz_total = df_mes_ano.pivot(index="mes", columns="ano", values="total")
    # Garante 12 meses no eixo Y mesmo que algum mês esteja ausente.
    matriz_prev = matriz_prev.reindex(index=range(1, 13))
    matriz_total = matriz_total.reindex(index=range(1, 13))

    mascara_pouco_dado = matriz_total.fillna(0).lt(n_minimo)
    matriz_para_plot = matriz_prev.where(~mascara_pouco_dado)

    figura, eixo = plt.subplots(figsize=(1.2 * matriz_prev.shape[1] + 2, 5))
    # Fundo cinza para células marcadas — desenhado por baixo do heatmap
    # principal via `imshow` com a máscara invertida.
    eixo.imshow(
        mascara_pouco_dado.to_numpy().astype(float),
        cmap="Greys",
        aspect="auto",
        alpha=0.35,
    )
    imagem = eixo.imshow(
        matriz_para_plot.to_numpy(dtype=float),
        cmap="viridis",
        aspect="auto",
    )

    eixo.set_xticks(range(matriz_prev.shape[1]))
    eixo.set_xticklabels([str(c) for c in matriz_prev.columns])
    eixo.set_yticks(range(12))
    eixo.set_yticklabels([f"{m:02d}" for m in range(1, 13)])
    eixo.set_xlabel("Ano")
    eixo.set_ylabel("Mês")
    eixo.set_title(titulo)

    n_celulas = matriz_prev.shape[0] * matriz_prev.shape[1]
    if n_celulas <= 40:
        for i in range(matriz_prev.shape[0]):
            for j in range(matriz_prev.shape[1]):
                valor = matriz_prev.iat[i, j]
                if pd.isna(valor):
                    rotulo = "—"
                elif mascara_pouco_dado.iat[i, j]:
                    rotulo = f"{valor:.1%}*"
                else:
                    rotulo = f"{valor:.1%}"
                eixo.text(
                    j,
                    i,
                    rotulo,
                    ha="center",
                    va="center",
                    color="white" if not mascara_pouco_dado.iat[i, j] else "#2f3e50",
                    fontsize=8,
                )

    barra_cor = figura.colorbar(imagem, ax=eixo, shrink=0.85)
    barra_cor.set_label("Prevalência")
    figura.tight_layout()
    return figura


def figura_contagem_diaria_com_anomalias(
    df_calendario: pd.DataFrame,
    df_anomalias: pd.DataFrame,
    titulo: str = "Contagem diária de notícias com anomalias |z|>3",
) -> mfigure.Figure:
    """Série diária em cinza claro com pontos de anomalia destacados.

    Três camadas de pontos:
    - clássico-apenas: flagado por z clássico mas não robusto;
    - robusto-apenas: flagado por z robusto mas não clássico;
    - ambos: flagado pelas duas regras (mais forte).
    """
    if "data" not in df_calendario.columns or "total" not in df_calendario.columns:
        raise ValueError("df_calendario precisa ter colunas 'data' e 'total'")

    calendario = df_calendario.sort_values(by="data", kind="mergesort")
    datas = pd.to_datetime(calendario["data"]).to_numpy()
    totais = calendario["total"].to_numpy(dtype=float)

    figura, eixo = plt.subplots(figsize=(11, 4.5))
    eixo.plot(
        datas,
        totais,
        color="#9aa4b2",
        linewidth=0.6,
        alpha=0.8,
        label="Contagem diária",
    )

    if not df_anomalias.empty:
        anomalias = df_anomalias.copy()
        anomalias["data"] = pd.to_datetime(anomalias["data"])
        ambos = anomalias.loc[anomalias["flag_ambos"]]
        so_classico = anomalias.loc[anomalias["flag_classico"] & ~anomalias["flag_robusto"]]
        so_robusto = anomalias.loc[anomalias["flag_robusto"] & ~anomalias["flag_classico"]]

        if not so_classico.empty:
            eixo.scatter(
                so_classico["data"].to_numpy(),
                so_classico["total"].to_numpy(),
                color="#DD8452",
                s=30,
                zorder=3,
                label=f"Só clássico (n={len(so_classico)})",
            )
        if not so_robusto.empty:
            eixo.scatter(
                so_robusto["data"].to_numpy(),
                so_robusto["total"].to_numpy(),
                color="#55A868",
                s=30,
                zorder=3,
                label=f"Só robusto (n={len(so_robusto)})",
            )
        if not ambos.empty:
            eixo.scatter(
                ambos["data"].to_numpy(),
                ambos["total"].to_numpy(),
                color="#C44E52",
                s=40,
                marker="D",
                zorder=4,
                label=f"Ambos (n={len(ambos)})",
            )

    eixo.set_xlabel("Dia")
    eixo.set_ylabel("Notícias publicadas")
    eixo.set_title(titulo)
    eixo.grid(True, axis="y", alpha=0.3)
    eixo.legend(loc="best", frameon=False, fontsize=9)
    eixo.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    eixo.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    for rotulo in eixo.get_xticklabels():
        rotulo.set_rotation(45)
        rotulo.set_ha("right")
    figura.tight_layout()
    return figura


def figura_jaccard_mercado_top7(
    df_sobreposicao: pd.DataFrame,
    titulo: str = "Sobreposição lexical Jaccard: 'mercado' × top-7 categorias (top-30 tokens TF-IDF)",
) -> mfigure.Figure:
    """Barras horizontais do Jaccard entre mercado e cada outra categoria.

    Espera saída de `sobreposicao_top_tokens` com colunas `categoria_vs,
    jaccard_top30`. Ordena decrescente para leitura fácil.
    """
    if df_sobreposicao.empty:
        raise ValueError("df_sobreposicao está vazio")

    dados = df_sobreposicao.sort_values(
        by="jaccard_top30", ascending=True, kind="mergesort"
    )
    figura, eixo = plt.subplots(figsize=(8, max(3, 0.4 * len(dados) + 1.5)))
    eixo.barh(
        dados["categoria_vs"].astype(str).tolist(),
        dados["jaccard_top30"].to_numpy(dtype=float),
        color="#4C72B0",
    )
    for idx, valor in enumerate(dados["jaccard_top30"].to_numpy(dtype=float)):
        eixo.text(valor, idx, f" {valor:.3f}", va="center", fontsize=9)
    eixo.set_xlabel("Jaccard (top-30 tokens TF-IDF)")
    eixo.set_ylabel("Categoria comparada")
    eixo.set_title(titulo)
    eixo.set_xlim(0, max(0.05, float(dados["jaccard_top30"].max()) * 1.25))
    eixo.grid(True, axis="x", alpha=0.3)
    figura.tight_layout()
    return figura


def figura_comprimento_medio_texto_por_trimestre(
    df_estatisticas: pd.DataFrame,
    titulo: str = "Comprimento médio de 'text' (tokens) por trimestre — notícias 'mercado'",
) -> mfigure.Figure:
    """Linha do comprimento médio do campo `text` em tokens, por trimestre."""
    if df_estatisticas.empty:
        raise ValueError("df_estatisticas está vazio")

    dados = df_estatisticas.copy().sort_values(by="trimestre", kind="mergesort")
    rotulos = dados["trimestre"].astype(str).tolist()
    y = dados["comprimento_medio_texto_tokens"].to_numpy(dtype=float)

    figura, eixo = plt.subplots(figsize=(10, 4))
    eixo.plot(rotulos, y, color="#4C72B0", marker="o", linewidth=1.5, markersize=5)
    eixo.set_xlabel("Trimestre")
    eixo.set_ylabel("Tokens por notícia (média)")
    eixo.set_title(titulo)
    eixo.grid(True, axis="y", alpha=0.3)
    for rotulo in eixo.get_xticklabels():
        rotulo.set_rotation(45)
        rotulo.set_ha("right")
    figura.tight_layout()
    return figura


def figura_psi_trimestres(
    df_psi: pd.DataFrame,
    titulo: str = "PSI top-200 vs. trimestre-baseline — notícias 'mercado'",
) -> mfigure.Figure:
    """Barras do PSI por trimestre em relação ao baseline.

    Espera DataFrame com colunas `trimestre, psi_top200`. Desenha linhas
    de referência em PSI = 0,1 (deriva leve) e 0,25 (deriva forte), que
    são os cortes usuais na literatura de *population stability*.
    """
    if df_psi.empty:
        raise ValueError("df_psi está vazio")

    dados = df_psi.copy().sort_values(by="trimestre", kind="mergesort")
    rotulos = dados["trimestre"].astype(str).tolist()
    y = dados["psi_top200"].to_numpy(dtype=float)

    figura, eixo = plt.subplots(figsize=(10, 4))
    cores = [
        "#55A868" if valor < 0.1 else ("#DD8452" if valor < 0.25 else "#C44E52")
        for valor in y
    ]
    eixo.bar(rotulos, y, color=cores)
    for idx, valor in enumerate(y):
        eixo.text(idx, valor, f"{valor:.3f}", ha="center", va="bottom", fontsize=8)
    eixo.axhline(0.10, color="#DD8452", linestyle="--", linewidth=1.0, label="PSI = 0.10 (leve)")
    eixo.axhline(0.25, color="#C44E52", linestyle="--", linewidth=1.0, label="PSI = 0.25 (forte)")
    eixo.set_xlabel("Trimestre")
    eixo.set_ylabel("PSI (vs. baseline)")
    eixo.set_title(titulo)
    eixo.grid(True, axis="y", alpha=0.3)
    eixo.legend(loc="best", frameon=False, fontsize=9)
    for rotulo in eixo.get_xticklabels():
        rotulo.set_rotation(45)
        rotulo.set_ha("right")
    figura.tight_layout()
    return figura


def _coerce_serie_para_indice_temporal(serie: pd.DataFrame) -> pd.DataFrame:
    """Aceita série com índice temporal OU coluna `data` e devolve índice temporal.

    As funções de figura querem um DatetimeIndex; as tabelas persistidas em
    CSV (contrato do ciclo: colunas `data, total, positivos, prevalencia`)
    vêm sem índice. Esse adaptador absorve as duas formas.
    """
    if isinstance(serie.index, pd.DatetimeIndex):
        return serie
    if "data" not in serie.columns:
        raise ValueError(
            "série precisa ter DatetimeIndex ou coluna 'data'"
        )
    saida = serie.copy()
    saida["data"] = pd.to_datetime(saida["data"])
    return saida.set_index("data").sort_index(kind="mergesort")
