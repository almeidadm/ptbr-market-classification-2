"""
Executa um eixo da EDA e persiste os artefatos em `artifacts/eda/`.

Uso:
    python scripts/executar_eda_secao.py --secao 1

Este script é idempotente: sobrescreve os artefatos da seção indicada.
Os caminhos de entrada e saída vêm de variáveis de ambiente para
funcionar em Colab sem modificação (CLAUDE.md §5):
    CAMINHO_CSV          default: data/raw/articles.csv
    DIRETORIO_ARTEFATOS  default: artifacts/eda
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

RAIZ_PROJETO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ_PROJETO))

import pandas as pd

from src.eda.anomalias import (
    detectar_anomalias_diarias,
    detectar_dias_zero,
    gerar_calendario,
)
from src.eda.carregamento import carregar_articles, eh_mercado
from src.eda.contagens import (
    intervalo_wilson,
    prevalencia_por_dia_semana,
    prevalencia_por_mes_ano,
    serie_total_e_classe,
)
from src.eda.duplicatas import (
    encontrar_near_duplicates,
    notificar_afetadas_exatas,
    notificar_afetadas_near,
    resumo_duplicatas_exatas,
)
from src.eda.figuras import (
    figura_contagem_diaria_com_anomalias,
    figura_distribuicao_categorias,
    figura_heatmap_prevalencia_mes_ano,
    figura_jaccard_mercado_top7,
    figura_prevalencia_dia_semana,
    figura_psi_trimestres,
    figura_comprimento_medio_texto_por_trimestre,
    figura_serie_diaria_prevalencia,
    figura_serie_mensal_contagens,
    figura_serie_mensal_prevalencia,
    figura_serie_semanal_prevalencia,
)
from src.eda.testes_temporais import (
    mann_kendall,
    qui_quadrado_k_por_2,
    z_proporcoes,
)
from src.eda.vocabulario import (
    REGEX_TOKEN_TFIDF,
    _carregar_stopwords_pt,
    jaccard_conjuntos,
    psi,
    sobreposicao_top_tokens,
    tokens_top_k_tfidf,
)

SEED = 20260423

# Decisão 003: out/2017 tem apenas 1 dia (2017-10-01) e é excluído da
# agregação mensal para não distorcer a série e o Mann-Kendall.
DATA_LIMITE_MENSAL = pd.Timestamp("2017-10-01")


def resolver_caminhos() -> tuple[Path, Path]:
    csv = Path(os.environ.get("CAMINHO_CSV", RAIZ_PROJETO / "data" / "raw" / "articles.csv"))
    artefatos = Path(os.environ.get("DIRETORIO_ARTEFATOS", RAIZ_PROJETO / "artifacts" / "eda"))
    return csv, artefatos


def executar_secao_1(caminho_csv: Path, diretorio_artefatos: Path) -> dict:
    """§1 — Volume e prevalência global, nulos, top-7 categorias.

    Persiste:
      - tabelas/01-cardinalidades-e-nulos.csv
      - tabelas/01-value-counts-category-top7.csv
      - tabelas/01-prevalencia-global.csv
      - figuras/01-distribuicao-categorias.{png,svg}

    Retorna dicionário com os principais números para exibição no notebook
    e para o relatório de parada após §1.
    """
    dir_tabelas = diretorio_artefatos / "tabelas"
    dir_figuras = diretorio_artefatos / "figuras"
    dir_tabelas.mkdir(parents=True, exist_ok=True)
    dir_figuras.mkdir(parents=True, exist_ok=True)

    df = carregar_articles(caminho_csv)
    total_linhas = len(df)

    nulos_por_coluna = df.isna().sum().astype("int64")
    cardinalidades = df.nunique(dropna=True).astype("int64")
    tabela_card_nulos = pd.DataFrame(
        {
            "coluna": df.columns,
            "nulos": [int(nulos_por_coluna[c]) for c in df.columns],
            "pct_nulos": [float(nulos_por_coluna[c]) / total_linhas for c in df.columns],
            "valores_unicos_nao_nulos": [int(cardinalidades[c]) for c in df.columns],
            "dtype": [str(df[c].dtype) for c in df.columns],
        }
    )
    tabela_card_nulos.to_csv(
        dir_tabelas / "01-cardinalidades-e-nulos.csv",
        index=False,
        encoding="utf-8",
    )

    contagem_categorias = df["category"].value_counts(dropna=False)
    top_7 = contagem_categorias.head(7)
    restante = int(contagem_categorias.iloc[7:].sum()) if contagem_categorias.size > 7 else 0
    tabela_top7 = pd.DataFrame(
        {
            "categoria": list(top_7.index.astype(str)) + (["outras"] if restante else []),
            "contagem": list(map(int, top_7.values)) + ([restante] if restante else []),
        }
    )
    tabela_top7["pct"] = tabela_top7["contagem"] / total_linhas
    tabela_top7.to_csv(
        dir_tabelas / "01-value-counts-category-top7.csv",
        index=False,
        encoding="utf-8",
    )

    mascara_mercado = eh_mercado(df["category"])
    positivos_total = int(mascara_mercado.sum())
    prevalencia, wilson_inf, wilson_sup = intervalo_wilson(positivos_total, total_linhas)

    linhas_sem_category = int(df["category"].isna().sum())
    linhas_sem_date = int(df["date"].isna().sum())

    tabela_prev = pd.DataFrame(
        [
            {
                "total_linhas": total_linhas,
                "positivos_mercado": positivos_total,
                "prevalencia": prevalencia,
                "wilson_ic95_inf": wilson_inf,
                "wilson_ic95_sup": wilson_sup,
                "linhas_com_category_nula": linhas_sem_category,
                "linhas_com_date_nula": linhas_sem_date,
            }
        ]
    )
    tabela_prev.to_csv(
        dir_tabelas / "01-prevalencia-global.csv",
        index=False,
        encoding="utf-8",
    )

    figura = figura_distribuicao_categorias(contagem_categorias.dropna(), top_n=7)
    figura.savefig(dir_figuras / "01-distribuicao-categorias.png", dpi=300, bbox_inches="tight")
    figura.savefig(dir_figuras / "01-distribuicao-categorias.svg", bbox_inches="tight")

    return {
        "total_linhas": total_linhas,
        "positivos_mercado": positivos_total,
        "prevalencia": prevalencia,
        "wilson_ic95": (wilson_inf, wilson_sup),
        "linhas_com_category_nula": linhas_sem_category,
        "linhas_com_date_nula": linhas_sem_date,
        "tabela_cardinalidades_nulos": tabela_card_nulos,
        "tabela_top7": tabela_top7,
        "tabela_prevalencia_global": tabela_prev,
    }


def _serie_para_csv(serie: pd.DataFrame) -> pd.DataFrame:
    """Converte `DataFrame` com índice temporal em frame com coluna `data`.

    Contrato do ciclo atual: tabelas de série gravadas com colunas
    `data, total, positivos, prevalencia` e `data` em ISO `YYYY-MM-DD`.
    Pandas entrega `DatetimeIndex` apontando para o final do período
    quando se usa `ME`/`W`; convertemos para o início do período via
    `to_period(...).to_timestamp()` para produzir uma data representativa
    e estável ("primeira data do período").
    """
    frame = serie.copy()
    indice = frame.index
    if isinstance(indice, pd.DatetimeIndex):
        if indice.freqstr == "ME":
            datas = indice.to_period("M").to_timestamp(how="start")
        elif indice.freqstr and indice.freqstr.startswith("W"):
            datas = indice.to_period("W").to_timestamp(how="start")
        else:
            datas = indice
        datas_iso = pd.DatetimeIndex(datas).strftime("%Y-%m-%d")
    else:
        datas_iso = indice.astype(str)

    frame = frame.reset_index(drop=True)
    frame.insert(0, "data", datas_iso)
    return frame[["data", "total", "positivos", "prevalencia"]]


def _cobertura_temporal(df: pd.DataFrame, serie_mensal: pd.DataFrame) -> pd.DataFrame:
    """Retorna tabela de cobertura temporal com 1 linha (alinhada ao contrato do ciclo).

    Inclui cobertura física (1.005 dias calendário entre `data_min` e `data_max`,
    987 com pelo menos uma publicação) e cobertura mensal efetiva conforme
    decisão 003 (2015-01 a 2017-09 = 33 meses completos). A série mensal
    recebida aqui já vem filtrada por out/2017.
    """
    datas_validas = df.loc[df["date"].notna(), "date"]
    data_min = datas_validas.min()
    data_max = datas_validas.max()
    dias_distintos = int(datas_validas.dt.normalize().nunique())
    dias_calendario = int((data_max.normalize() - data_min.normalize()).days) + 1

    meses_zerados = serie_mensal.loc[serie_mensal["total"] == 0]
    periodos_zerados_iso = [
        ts.to_period("M").to_timestamp(how="start").strftime("%Y-%m")
        for ts in meses_zerados.index
    ]

    mes_mensal_min = serie_mensal.index.min().to_period("M").strftime("%Y-%m")
    mes_mensal_max = serie_mensal.index.max().to_period("M").strftime("%Y-%m")
    cobertura_mensal_efetiva = f"{mes_mensal_min} a {mes_mensal_max}"

    return pd.DataFrame(
        [
            {
                "data_min": data_min.strftime("%Y-%m-%d"),
                "data_max": data_max.strftime("%Y-%m-%d"),
                "dias_distintos": dias_distintos,
                "dias_calendario": dias_calendario,
                "meses_zerados": json.dumps(periodos_zerados_iso, ensure_ascii=False),
                "n_meses_zerados": len(periodos_zerados_iso),
                "n_pontos_serie_mensal": int(len(serie_mensal)),
                "cobertura_mensal_efetiva": cobertura_mensal_efetiva,
            }
        ]
    )


def executar_secao_2(caminho_csv: Path, diretorio_artefatos: Path) -> dict:
    """§2 — Série temporal da classe (mensal principal; semanal e diária no apêndice).

    Persiste:
      - tabelas/02-serie-{mensal,semanal,diaria}.csv
      - tabelas/02-cobertura-temporal.csv
      - figuras/02-serie-mensal-prevalencia.{png,svg}
      - figuras/02-serie-mensal-contagens.{png,svg}
      - figuras/02-serie-semanal-prevalencia.{png,svg}
      - figuras/02-serie-diaria-prevalencia.{png,svg}

    Retorna dicionário com as três séries (indexadas por tempo) e a
    tabela de cobertura — reutilizáveis por §3 sem recomputar.
    """
    dir_tabelas = diretorio_artefatos / "tabelas"
    dir_figuras = diretorio_artefatos / "figuras"
    dir_tabelas.mkdir(parents=True, exist_ok=True)
    dir_figuras.mkdir(parents=True, exist_ok=True)

    df = carregar_articles(caminho_csv)
    df = df.assign(is_mercado=eh_mercado(df["category"]))

    # Decisão 003: exclui out/2017 (n=121, 1 único dia) da agregação mensal
    # para não distorcer a série e o Mann-Kendall. Séries diária e semanal
    # continuam sobre todo o corpus — o gap aparece nelas por natureza.
    df_mensal = df.loc[df["date"] < DATA_LIMITE_MENSAL]
    serie_mensal = serie_total_e_classe(df_mensal, "is_mercado", "M")
    serie_semanal = serie_total_e_classe(df, "is_mercado", "W")
    serie_diaria = serie_total_e_classe(df, "is_mercado", "D")

    _serie_para_csv(serie_mensal).to_csv(
        dir_tabelas / "02-serie-mensal.csv", index=False, encoding="utf-8"
    )
    _serie_para_csv(serie_semanal).to_csv(
        dir_tabelas / "02-serie-semanal.csv", index=False, encoding="utf-8"
    )
    _serie_para_csv(serie_diaria).to_csv(
        dir_tabelas / "02-serie-diaria.csv", index=False, encoding="utf-8"
    )

    cobertura = _cobertura_temporal(df, serie_mensal)
    cobertura.to_csv(
        dir_tabelas / "02-cobertura-temporal.csv", index=False, encoding="utf-8"
    )

    figura_prev = figura_serie_mensal_prevalencia(serie_mensal)
    figura_prev.savefig(
        dir_figuras / "02-serie-mensal-prevalencia.png", dpi=300, bbox_inches="tight"
    )
    figura_prev.savefig(
        dir_figuras / "02-serie-mensal-prevalencia.svg", bbox_inches="tight"
    )

    figura_cont = figura_serie_mensal_contagens(serie_mensal)
    figura_cont.savefig(
        dir_figuras / "02-serie-mensal-contagens.png", dpi=300, bbox_inches="tight"
    )
    figura_cont.savefig(
        dir_figuras / "02-serie-mensal-contagens.svg", bbox_inches="tight"
    )

    figura_sem = figura_serie_semanal_prevalencia(serie_semanal)
    figura_sem.savefig(
        dir_figuras / "02-serie-semanal-prevalencia.png", dpi=300, bbox_inches="tight"
    )
    figura_sem.savefig(
        dir_figuras / "02-serie-semanal-prevalencia.svg", bbox_inches="tight"
    )

    figura_dia = figura_serie_diaria_prevalencia(serie_diaria)
    figura_dia.savefig(
        dir_figuras / "02-serie-diaria-prevalencia.png", dpi=300, bbox_inches="tight"
    )
    figura_dia.savefig(
        dir_figuras / "02-serie-diaria-prevalencia.svg", bbox_inches="tight"
    )

    return {
        "df_com_alvo": df,
        "serie_mensal": serie_mensal,
        "serie_semanal": serie_semanal,
        "serie_diaria": serie_diaria,
        "cobertura": cobertura,
    }


def _tabela_ano_positivos_negativos(df: pd.DataFrame) -> pd.DataFrame:
    """Tabela com k linhas (k = anos distintos) e colunas `ano, positivos, negativos, total`.

    Descarta linhas com `date` nulo (sem ano atribuível). Ordenação por ano.
    """
    dados = df.loc[df["date"].notna(), ["date", "is_mercado"]].copy()
    dados["ano"] = dados["date"].dt.year.astype("int64")
    grupos = dados.groupby("ano", sort=True)
    positivos = grupos["is_mercado"].sum().astype("int64").rename("positivos")
    total = grupos.size().rename("total").astype("int64")
    frame = pd.concat([positivos, total], axis=1).reset_index()
    frame["negativos"] = frame["total"] - frame["positivos"]
    return frame[["ano", "positivos", "negativos", "total"]]


def executar_secao_3(caminho_csv: Path, diretorio_artefatos: Path) -> dict:
    """§3 — Testes de deriva sobre a série mensal e blocos anuais.

    Depende logicamente de §2 (série mensal). Para evitar releitura do CSV
    quando §2 e §3 rodam na mesma invocação, o orquestrador pode passar
    pré-computado via `main`. Quando chamado isoladamente, recomputa
    chamando `executar_secao_2`.

    Persiste:
      - tabelas/03-testes-deriva.csv
    """
    saida_2 = executar_secao_2(caminho_csv, diretorio_artefatos)
    return _executar_secao_3_a_partir_de(saida_2, diretorio_artefatos)


def _executar_secao_3_a_partir_de(
    saida_secao_2: dict,
    diretorio_artefatos: Path,
) -> dict:
    dir_tabelas = diretorio_artefatos / "tabelas"
    dir_tabelas.mkdir(parents=True, exist_ok=True)

    df = saida_secao_2["df_com_alvo"]
    serie_mensal = saida_secao_2["serie_mensal"]

    resultados = []

    mk = mann_kendall(serie_mensal["prevalencia"].to_numpy())
    resultados.append(mk)

    tabela_anos = _tabela_ano_positivos_negativos(df)
    anos = tabela_anos["ano"].tolist()

    if len(anos) >= 2:
        primeiro = tabela_anos.iloc[0]
        ultimo = tabela_anos.iloc[-1]
        zp = z_proporcoes(
            n1=int(primeiro["total"]),
            k1=int(primeiro["positivos"]),
            n2=int(ultimo["total"]),
            k2=int(ultimo["positivos"]),
        )
        zp = _anexar_anos_ao_efeito(zp, int(primeiro["ano"]), int(ultimo["ano"]))
        resultados.append(zp)
    else:
        resultados.append(_resultado_pulado("z_proporcoes", anos))

    if len(anos) >= 2:
        tabela_qui = tabela_anos[["positivos", "negativos"]].to_numpy()
        qui = qui_quadrado_k_por_2(tabela_qui)
        qui = _anexar_anos_ao_efeito(qui, None, None, todos_anos=anos)
        resultados.append(qui)
    else:
        resultados.append(_resultado_pulado("qui_quadrado_k_por_2", anos))

    tabela = pd.DataFrame(
        [
            {
                "nome": r.nome,
                "estatistica": r.estatistica,
                "p_valor": r.p_valor,
                "efeito": json.dumps(r.efeito, ensure_ascii=False),
            }
            for r in resultados
        ]
    )
    tabela.to_csv(
        dir_tabelas / "03-testes-deriva.csv", index=False, encoding="utf-8"
    )

    return {
        "resultados": resultados,
        "tabela_testes": tabela,
        "tabela_anos": tabela_anos,
    }


def _anexar_anos_ao_efeito(
    resultado,
    ano_1,
    ano_2,
    todos_anos=None,
):
    """Anexa metadados de anos ao dicionário `efeito` (versão imutável do dataclass)."""
    from dataclasses import replace

    novo_efeito = dict(resultado.efeito)
    if ano_1 is not None:
        novo_efeito["ano_1"] = ano_1
    if ano_2 is not None:
        novo_efeito["ano_2"] = ano_2
    if todos_anos is not None:
        novo_efeito["anos"] = list(todos_anos)
    return replace(resultado, efeito=novo_efeito)


def _resultado_pulado(nome: str, anos):
    from src.eda.testes_temporais import ResultadoTeste

    return ResultadoTeste(
        nome=nome,
        estatistica=float("nan"),
        p_valor=float("nan"),
        efeito={"motivo": "corpus com <2 anos distintos", "anos": list(anos)},
    )


def executar_secao_4(caminho_csv: Path, diretorio_artefatos: Path) -> dict:
    """§4 — Sazonalidade: prevalência por dia-da-semana e heatmap mês×ano.

    Persiste:
      - tabelas/04-prevalencia-por-dia-semana.csv
      - tabelas/04-prevalencia-mes-ano.csv
      - figuras/04-prevalencia-dia-semana.{png,svg}
      - figuras/04-heatmap-prevalencia-mes-ano.{png,svg}
    """
    dir_tabelas = diretorio_artefatos / "tabelas"
    dir_figuras = diretorio_artefatos / "figuras"
    dir_tabelas.mkdir(parents=True, exist_ok=True)
    dir_figuras.mkdir(parents=True, exist_ok=True)

    df = carregar_articles(caminho_csv)
    df = df.assign(is_mercado=eh_mercado(df["category"]))

    tabela_dia_semana = prevalencia_por_dia_semana(df, "is_mercado")
    tabela_mes_ano = prevalencia_por_mes_ano(df, "is_mercado")

    tabela_dia_semana.to_csv(
        dir_tabelas / "04-prevalencia-por-dia-semana.csv",
        index=False,
        encoding="utf-8",
    )
    tabela_mes_ano.to_csv(
        dir_tabelas / "04-prevalencia-mes-ano.csv",
        index=False,
        encoding="utf-8",
    )

    # Prevalência global para a linha de referência da figura de dia-da-semana.
    mascara_mercado = df["is_mercado"].astype(bool)
    total_linhas_validas = int(df["date"].notna().sum())
    prevalencia_global = (
        float(mascara_mercado[df["date"].notna()].sum()) / total_linhas_validas
        if total_linhas_validas > 0
        else float("nan")
    )

    figura_dia = figura_prevalencia_dia_semana(
        tabela_dia_semana, prevalencia_global=prevalencia_global
    )
    figura_dia.savefig(
        dir_figuras / "04-prevalencia-dia-semana.png", dpi=300, bbox_inches="tight"
    )
    figura_dia.savefig(
        dir_figuras / "04-prevalencia-dia-semana.svg", bbox_inches="tight"
    )

    figura_heat = figura_heatmap_prevalencia_mes_ano(tabela_mes_ano)
    figura_heat.savefig(
        dir_figuras / "04-heatmap-prevalencia-mes-ano.png", dpi=300, bbox_inches="tight"
    )
    figura_heat.savefig(
        dir_figuras / "04-heatmap-prevalencia-mes-ano.svg", bbox_inches="tight"
    )

    return {
        "tabela_dia_semana": tabela_dia_semana,
        "tabela_mes_ano": tabela_mes_ano,
        "prevalencia_global": prevalencia_global,
    }


def executar_secao_5(caminho_csv: Path, diretorio_artefatos: Path) -> dict:
    """§5 — Lacunas e anomalias na série diária de contagem total.

    Persiste:
      - tabelas/05-dias-zero.csv
      - tabelas/05-anomalias-diarias.csv
      - figuras/05-contagem-diaria-com-anomalias.{png,svg}
    """
    dir_tabelas = diretorio_artefatos / "tabelas"
    dir_figuras = diretorio_artefatos / "figuras"
    dir_tabelas.mkdir(parents=True, exist_ok=True)
    dir_figuras.mkdir(parents=True, exist_ok=True)

    df = carregar_articles(caminho_csv)
    calendario = gerar_calendario(df)
    dias_zero = detectar_dias_zero(calendario)
    anomalias = detectar_anomalias_diarias(calendario, limiar=3.0)

    dias_zero_csv = dias_zero.copy()
    if not dias_zero_csv.empty:
        dias_zero_csv["data"] = pd.to_datetime(dias_zero_csv["data"]).dt.strftime("%Y-%m-%d")
    dias_zero_csv.to_csv(
        dir_tabelas / "05-dias-zero.csv", index=False, encoding="utf-8"
    )

    anomalias_csv = anomalias.copy()
    if not anomalias_csv.empty:
        anomalias_csv["data"] = pd.to_datetime(anomalias_csv["data"]).dt.strftime("%Y-%m-%d")
    anomalias_csv.to_csv(
        dir_tabelas / "05-anomalias-diarias.csv", index=False, encoding="utf-8"
    )

    figura = figura_contagem_diaria_com_anomalias(calendario, anomalias)
    figura.savefig(
        dir_figuras / "05-contagem-diaria-com-anomalias.png", dpi=300, bbox_inches="tight"
    )
    figura.savefig(
        dir_figuras / "05-contagem-diaria-com-anomalias.svg", bbox_inches="tight"
    )

    return {
        "calendario": calendario,
        "dias_zero": dias_zero,
        "anomalias": anomalias,
    }


def executar_secao_6(caminho_csv: Path, diretorio_artefatos: Path) -> dict:
    """§6 — Co-ocorrência lexical entre `mercado` e as top-7 demais categorias.

    Identifica programaticamente as 7 maiores categorias (por contagem total),
    aplica TF-IDF com stopwords PT-BR sobre amostra estratificada determinística
    de até 2.000 notícias por categoria, extrai top-30 tokens por categoria
    e mede Jaccard / sobreposição contra o top-30 de `mercado`.

    Persiste:
      - tabelas/06-top-tokens-por-categoria.csv
      - tabelas/06-sobreposicao-mercado-vs-top7.csv
      - figuras/06-jaccard-mercado-vs-top7.{png,svg}
    """
    dir_tabelas = diretorio_artefatos / "tabelas"
    dir_figuras = diretorio_artefatos / "figuras"
    dir_tabelas.mkdir(parents=True, exist_ok=True)
    dir_figuras.mkdir(parents=True, exist_ok=True)

    df = carregar_articles(caminho_csv)

    # Top-7 categorias diferentes de 'mercado', + mercado -> 8 categorias total.
    contagem_cat = df["category"].value_counts(dropna=True)
    top7_exceto_mercado = [
        str(categoria)
        for categoria in contagem_cat.index
        if str(categoria) != "mercado"
    ][:7]
    categorias_analisadas = ["mercado", *top7_exceto_mercado]

    top_tokens = tokens_top_k_tfidf(
        df=df,
        coluna_categoria="category",
        coluna_texto="text",
        categorias=categorias_analisadas,
        k=30,
        n_amostra_por_categoria=2000,
        seed=SEED,
    )
    top_tokens.to_csv(
        dir_tabelas / "06-top-tokens-por-categoria.csv",
        index=False,
        encoding="utf-8",
    )

    sobreposicao = sobreposicao_top_tokens(top_tokens, categoria_referencia="mercado")
    sobreposicao.to_csv(
        dir_tabelas / "06-sobreposicao-mercado-vs-top7.csv",
        index=False,
        encoding="utf-8",
    )

    figura = figura_jaccard_mercado_top7(sobreposicao)
    figura.savefig(
        dir_figuras / "06-jaccard-mercado-vs-top7.png", dpi=300, bbox_inches="tight"
    )
    figura.savefig(
        dir_figuras / "06-jaccard-mercado-vs-top7.svg", bbox_inches="tight"
    )

    return {
        "categorias_analisadas": categorias_analisadas,
        "top_tokens": top_tokens,
        "sobreposicao": sobreposicao,
    }


def executar_secao_7(
    caminho_csv: Path,
    diretorio_artefatos: Path,
    n_processos: int | None = None,
) -> dict:
    """§7 — Duplicatas exatas (link/title/text) + near-duplicates via MinHash/LSH.

    Decisão local: linhas com `text` nulo são descartadas só aqui (§7 e §8),
    não no corpus bruto. A §7 reporta fração do corpus (denominador = total
    bruto) afetada por duplicatas, e dispara parada se > 1%.

    `n_processos` é forwardado para `encontrar_near_duplicates` e controla
    a paralelização da construção de MinHashes. `None` → `os.cpu_count() - 1`.

    Persiste:
      - tabelas/07-duplicatas-exatas-resumo.csv
      - tabelas/07-near-duplicates-pares.csv
      - tabelas/07-near-duplicates-inter-periodo.csv
      - tabelas/07-resumo-deduplicacao.csv
    """
    dir_tabelas = diretorio_artefatos / "tabelas"
    dir_tabelas.mkdir(parents=True, exist_ok=True)

    df_bruto = carregar_articles(caminho_csv)
    total_bruto = len(df_bruto)

    df = df_bruto.loc[df_bruto["text"].notna()].copy()
    print(
        f"[§7] corpus bruto: {total_bruto:,} linhas; após remover text nulo: {len(df):,}".replace(",", ".")
    )

    resumo_exatas = resumo_duplicatas_exatas(df_bruto, ["link", "title", "text"])
    resumo_exatas.to_csv(
        dir_tabelas / "07-duplicatas-exatas-resumo.csv",
        index=False,
        encoding="utf-8",
    )

    pares = encontrar_near_duplicates(df, coluna_texto="text", n_processos=n_processos)

    if not pares.empty:
        metadados = df_bruto[["title", "date", "category"]]
        pares_expandidos = pares.join(metadados.add_suffix("_a"), on="id_a")
        pares_expandidos = pares_expandidos.join(metadados.add_suffix("_b"), on="id_b")
        pares_expandidos["date_a"] = pd.to_datetime(pares_expandidos["date_a"]).dt.strftime("%Y-%m-%d")
        pares_expandidos["date_b"] = pd.to_datetime(pares_expandidos["date_b"]).dt.strftime("%Y-%m-%d")
        pares_expandidos = pares_expandidos[
            [
                "id_a",
                "id_b",
                "jaccard_estimado",
                "title_a",
                "title_b",
                "date_a",
                "date_b",
                "category_a",
                "category_b",
            ]
        ]
    else:
        pares_expandidos = pd.DataFrame(
            columns=[
                "id_a",
                "id_b",
                "jaccard_estimado",
                "title_a",
                "title_b",
                "date_a",
                "date_b",
                "category_a",
                "category_b",
            ]
        )

    # Truncamento defensivo — CSVs gigantes quebram o repositório.
    truncado = False
    if len(pares_expandidos) > 10000:
        truncado = True
        pares_expandidos = pares_expandidos.head(10000)

    pares_expandidos.to_csv(
        dir_tabelas / "07-near-duplicates-pares.csv",
        index=False,
        encoding="utf-8",
    )

    # Pares com distância temporal > 30 dias: contaminam split temporal.
    inter_periodo = pd.DataFrame(columns=pares_expandidos.columns)
    if not pares_expandidos.empty:
        data_a = pd.to_datetime(pares_expandidos["date_a"], errors="coerce")
        data_b = pd.to_datetime(pares_expandidos["date_b"], errors="coerce")
        mascara_distantes = (data_a - data_b).abs() > pd.Timedelta(days=30)
        inter_periodo = pares_expandidos.loc[mascara_distantes].copy()
    inter_periodo.to_csv(
        dir_tabelas / "07-near-duplicates-inter-periodo.csv",
        index=False,
        encoding="utf-8",
    )

    afetadas_exatas = notificar_afetadas_exatas(df_bruto, ["link", "title", "text"])
    afetadas_near = notificar_afetadas_near(pares[["id_a", "id_b"]]) if not pares.empty else set()
    afetadas_uniao = afetadas_exatas | afetadas_near
    pct_afetadas = len(afetadas_uniao) / total_bruto if total_bruto > 0 else 0.0
    parada_acionada = pct_afetadas > 0.01

    resumo_dedupe = pd.DataFrame(
        [
            {
                "total_corpus_bruto": total_bruto,
                "linhas_com_text_nulo": int(df_bruto["text"].isna().sum()),
                "afetadas_exatas": len(afetadas_exatas),
                "afetadas_near": len(afetadas_near),
                "afetadas_uniao": len(afetadas_uniao),
                "pct_corpus_afetado": pct_afetadas,
                "parada_acionada": bool(parada_acionada),
                "pares_near_encontrados": int(len(pares)),
                "pares_inter_periodo": int(len(inter_periodo)),
                "pares_truncados_no_csv": bool(truncado),
            }
        ]
    )
    resumo_dedupe.to_csv(
        dir_tabelas / "07-resumo-deduplicacao.csv",
        index=False,
        encoding="utf-8",
    )

    return {
        "resumo_exatas": resumo_exatas,
        "pares": pares_expandidos,
        "inter_periodo": inter_periodo,
        "resumo_dedupe": resumo_dedupe,
        "parada_acionada": parada_acionada,
        "pct_afetadas": pct_afetadas,
    }


def _tokenizar_trimestre(texto: str, stopwords_set: frozenset[str]) -> list[str]:
    """Tokeniza com o mesmo regex da §6, aplicando stopwords PT-BR."""
    import re

    if not isinstance(texto, str):
        return []
    tokens = re.findall(REGEX_TOKEN_TFIDF, texto.lower())
    return [t for t in tokens if t not in stopwords_set]


def executar_secao_8(caminho_csv: Path, diretorio_artefatos: Path) -> dict:
    """§8 — Deriva mínima de vocabulário por trimestre (classe mercado).

    Produz comprimento médio de `title` e `text` (chars e tokens) por
    trimestre; Jaccard top-50 entre trimestres consecutivos; PSI sobre
    bag-of-top-200-tokens do corpus mercado (baseline = trimestre mais
    antigo).

    Persiste:
      - tabelas/08-estatisticas-trimestre.csv
      - tabelas/08-top50-tokens-por-trimestre.csv
      - tabelas/08-jaccard-trimestres-consecutivos.csv
      - tabelas/08-psi-trimestre-vs-baseline.csv
      - figuras/08-comprimento-medio-texto-por-trimestre.{png,svg}
      - figuras/08-psi-trimestres.{png,svg}
    """
    from collections import Counter

    dir_tabelas = diretorio_artefatos / "tabelas"
    dir_figuras = diretorio_artefatos / "figuras"
    dir_tabelas.mkdir(parents=True, exist_ok=True)
    dir_figuras.mkdir(parents=True, exist_ok=True)

    df = carregar_articles(caminho_csv)
    df = df.assign(is_mercado=eh_mercado(df["category"]))
    df_mercado = df.loc[df["is_mercado"] & df["text"].notna() & df["date"].notna()].copy()
    df_mercado["trimestre"] = df_mercado["date"].dt.to_period("Q").astype(str)

    stopwords_set = _carregar_stopwords_pt()

    # 1. Top-50 tokens por trimestre + comprimentos médios.
    estatisticas_linhas = []
    top50_linhas = []
    tokens_por_trimestre: dict[str, Counter] = {}
    trimestres_ordenados = sorted(df_mercado["trimestre"].unique())

    for trimestre in trimestres_ordenados:
        sub = df_mercado.loc[df_mercado["trimestre"] == trimestre]
        contador: Counter[str] = Counter()
        tokens_por_texto: list[int] = []
        chars_por_texto: list[int] = []
        tokens_por_titulo: list[int] = []
        chars_por_titulo: list[int] = []
        for texto, titulo in zip(sub["text"].to_list(), sub["title"].to_list()):
            tokens_texto = _tokenizar_trimestre(texto, stopwords_set)
            tokens_titulo = _tokenizar_trimestre(titulo, stopwords_set)
            contador.update(tokens_texto)
            tokens_por_texto.append(len(tokens_texto))
            chars_por_texto.append(len(texto) if isinstance(texto, str) else 0)
            tokens_por_titulo.append(len(tokens_titulo))
            chars_por_titulo.append(len(titulo) if isinstance(titulo, str) else 0)

        tokens_por_trimestre[trimestre] = contador

        estatisticas_linhas.append(
            {
                "trimestre": trimestre,
                "n_noticias": int(len(sub)),
                "comprimento_medio_titulo_chars": (
                    float(sum(chars_por_titulo) / len(chars_por_titulo))
                    if chars_por_titulo else 0.0
                ),
                "comprimento_medio_titulo_tokens": (
                    float(sum(tokens_por_titulo) / len(tokens_por_titulo))
                    if tokens_por_titulo else 0.0
                ),
                "comprimento_medio_texto_chars": (
                    float(sum(chars_por_texto) / len(chars_por_texto))
                    if chars_por_texto else 0.0
                ),
                "comprimento_medio_texto_tokens": (
                    float(sum(tokens_por_texto) / len(tokens_por_texto))
                    if tokens_por_texto else 0.0
                ),
            }
        )

        for rank, (token, freq) in enumerate(contador.most_common(50), start=1):
            top50_linhas.append(
                {"trimestre": trimestre, "rank": rank, "token": token, "freq": int(freq)}
            )

    estatisticas = pd.DataFrame(estatisticas_linhas)
    top50 = pd.DataFrame(top50_linhas)
    estatisticas.to_csv(
        dir_tabelas / "08-estatisticas-trimestre.csv", index=False, encoding="utf-8"
    )
    top50.to_csv(
        dir_tabelas / "08-top50-tokens-por-trimestre.csv", index=False, encoding="utf-8"
    )

    # 2. Jaccard top-50 entre trimestres consecutivos.
    jaccard_linhas = []
    for anterior, atual in zip(trimestres_ordenados[:-1], trimestres_ordenados[1:]):
        tokens_ant = {t for t, _ in tokens_por_trimestre[anterior].most_common(50)}
        tokens_atu = {t for t, _ in tokens_por_trimestre[atual].most_common(50)}
        jaccard_linhas.append(
            {
                "trimestre_anterior": anterior,
                "trimestre_atual": atual,
                "jaccard_top50": jaccard_conjuntos(tokens_ant, tokens_atu),
            }
        )
    jaccard_df = pd.DataFrame(jaccard_linhas)
    jaccard_df.to_csv(
        dir_tabelas / "08-jaccard-trimestres-consecutivos.csv",
        index=False,
        encoding="utf-8",
    )

    # 3. PSI sobre bag-of-top-200-tokens (do corpus completo mercado) — baseline = primeiro trimestre.
    contador_global: Counter[str] = Counter()
    for contador in tokens_por_trimestre.values():
        contador_global.update(contador)
    vocab_top200 = [token for token, _ in contador_global.most_common(200)]

    def distribuicao_top200(contador: Counter[str]) -> dict[str, float]:
        soma = sum(contador[t] for t in vocab_top200)
        if soma == 0:
            return {t: 0.0 for t in vocab_top200}
        return {t: contador[t] / soma for t in vocab_top200}

    trimestre_baseline = trimestres_ordenados[0]
    dist_baseline = distribuicao_top200(tokens_por_trimestre[trimestre_baseline])

    psi_linhas = []
    for trimestre in trimestres_ordenados:
        dist_atual = distribuicao_top200(tokens_por_trimestre[trimestre])
        valor_psi = psi(dist_baseline, dist_atual)
        psi_linhas.append({"trimestre": trimestre, "psi_top200": float(valor_psi)})
    psi_df = pd.DataFrame(psi_linhas)
    psi_df.to_csv(
        dir_tabelas / "08-psi-trimestre-vs-baseline.csv",
        index=False,
        encoding="utf-8",
    )

    figura_comp = figura_comprimento_medio_texto_por_trimestre(estatisticas)
    figura_comp.savefig(
        dir_figuras / "08-comprimento-medio-texto-por-trimestre.png",
        dpi=300,
        bbox_inches="tight",
    )
    figura_comp.savefig(
        dir_figuras / "08-comprimento-medio-texto-por-trimestre.svg",
        bbox_inches="tight",
    )

    figura_psi = figura_psi_trimestres(psi_df)
    figura_psi.savefig(
        dir_figuras / "08-psi-trimestres.png", dpi=300, bbox_inches="tight"
    )
    figura_psi.savefig(
        dir_figuras / "08-psi-trimestres.svg", bbox_inches="tight"
    )

    return {
        "estatisticas": estatisticas,
        "top50": top50,
        "jaccard": jaccard_df,
        "psi": psi_df,
        "trimestre_baseline": trimestre_baseline,
    }


SECOES = {
    1: executar_secao_1,
    2: executar_secao_2,
    3: executar_secao_3,
    4: executar_secao_4,
    5: executar_secao_5,
    6: executar_secao_6,
    7: executar_secao_7,
    8: executar_secao_8,
}


def _imprimir_sumario_secao_1(saida: dict) -> None:
    print(f"total_linhas              = {saida['total_linhas']:,}".replace(",", "."))
    print(f"positivos_mercado         = {saida['positivos_mercado']:,}".replace(",", "."))
    prev = saida["prevalencia"]
    baixo, alto = saida["wilson_ic95"]
    print(f"prevalencia               = {prev:.4%} (IC 95% Wilson: [{baixo:.4%}, {alto:.4%}])")
    print(f"linhas_com_category_nula  = {saida['linhas_com_category_nula']}")
    print(f"linhas_com_date_nula      = {saida['linhas_com_date_nula']}")
    print()
    print("Top-7 categorias + outras:")
    print(saida["tabela_top7"].to_string(index=False))
    print()
    print("Cardinalidades e nulos por coluna:")
    print(saida["tabela_cardinalidades_nulos"].to_string(index=False))


def _imprimir_sumario_secao_2(saida: dict) -> None:
    cobertura = saida["cobertura"].iloc[0]
    print("Cobertura temporal:")
    print(f"  data_min                  = {cobertura['data_min']}")
    print(f"  data_max                  = {cobertura['data_max']}")
    print(f"  dias_distintos            = {cobertura['dias_distintos']}")
    print(f"  dias_calendario           = {cobertura['dias_calendario']}")
    print(f"  n_meses_zerados           = {cobertura['n_meses_zerados']}")
    print(f"  meses_zerados             = {cobertura['meses_zerados']}")
    print(f"  n_pontos_serie_mensal     = {cobertura['n_pontos_serie_mensal']}")
    print(f"  cobertura_mensal_efetiva  = {cobertura['cobertura_mensal_efetiva']}")
    print()
    print("Série mensal (head):")
    print(saida["serie_mensal"].head().to_string())
    print()
    print("Série mensal (tail):")
    print(saida["serie_mensal"].tail().to_string())


def _imprimir_sumario_secao_3(saida: dict) -> None:
    print("Testes de deriva:")
    print(saida["tabela_testes"].to_string(index=False))
    print()
    print("Tabela ano × (positivos, negativos, total):")
    print(saida["tabela_anos"].to_string(index=False))


def _imprimir_sumario_secao_4(saida: dict) -> None:
    print("Prevalência por dia-da-semana:")
    print(saida["tabela_dia_semana"].to_string(index=False))
    print()
    print(f"Prevalência global (referência): {saida['prevalencia_global']:.4%}")
    print()
    print("Prevalência por mês×ano (head):")
    print(saida["tabela_mes_ano"].head(12).to_string(index=False))


def _imprimir_sumario_secao_5(saida: dict) -> None:
    dias_zero = saida["dias_zero"]
    anomalias = saida["anomalias"]
    print(f"Dias zero: {len(dias_zero)}")
    if not dias_zero.empty:
        print(dias_zero.to_string(index=False))
    print()
    flag_classico = int(anomalias["flag_classico"].sum()) if not anomalias.empty else 0
    flag_robusto = int(anomalias["flag_robusto"].sum()) if not anomalias.empty else 0
    flag_ambos = int(anomalias["flag_ambos"].sum()) if not anomalias.empty else 0
    print(
        f"Dias anômalos — clássico: {flag_classico}, "
        f"robusto: {flag_robusto}, ambos: {flag_ambos}"
    )
    if not anomalias.empty:
        print()
        print("Anomalias (head por data):")
        print(anomalias.head(10).to_string(index=False))


def _imprimir_sumario_secao_6(saida: dict) -> None:
    print(f"Categorias analisadas: {saida['categorias_analisadas']}")
    print()
    print("Sobreposição (top-30 TF-IDF) 'mercado' × outras categorias:")
    print(
        saida["sobreposicao"][["categoria_vs", "jaccard_top30"]]
        .to_string(index=False)
    )
    print()
    print("Top-10 tokens de 'mercado':")
    mercado = saida["top_tokens"].loc[saida["top_tokens"]["categoria"] == "mercado"]
    print(mercado.head(10).to_string(index=False))


def _imprimir_sumario_secao_7(saida: dict) -> None:
    print("Resumo duplicatas exatas:")
    print(
        saida["resumo_exatas"][["tipo", "linhas_duplicadas", "grupos", "pct_corpus"]]
        .to_string(index=False)
    )
    print()
    print("Resumo deduplicação:")
    print(saida["resumo_dedupe"].to_string(index=False))
    print()
    if saida["parada_acionada"]:
        print(
            "*** PARADA ACIONADA: duplicatas + near > 1% do corpus. "
            "§8 não será executada até que o usuário decida política de deduplicação. ***"
        )


def _imprimir_sumario_secao_8(saida: dict) -> None:
    print(f"Trimestre baseline (para PSI): {saida['trimestre_baseline']}")
    print()
    print("Estatísticas por trimestre:")
    print(saida["estatisticas"].to_string(index=False))
    print()
    print("Jaccard top-50 entre trimestres consecutivos:")
    print(saida["jaccard"].to_string(index=False))
    print()
    print("PSI vs. baseline:")
    print(saida["psi"].to_string(index=False))


SUMARIOS = {
    1: _imprimir_sumario_secao_1,
    2: _imprimir_sumario_secao_2,
    3: _imprimir_sumario_secao_3,
    4: _imprimir_sumario_secao_4,
    5: _imprimir_sumario_secao_5,
    6: _imprimir_sumario_secao_6,
    7: _imprimir_sumario_secao_7,
    8: _imprimir_sumario_secao_8,
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Executa uma seção da EDA (decisão 002).")
    parser.add_argument(
        "--secao",
        required=True,
        help="Número da seção (1 a 8) ou 'all' para rodar todas as seções implementadas.",
    )
    parser.add_argument(
        "--n-processos",
        type=int,
        default=None,
        help="Workers paralelos em §7 (construção de MinHashes). Default: os.cpu_count() - 1.",
    )
    args = parser.parse_args()

    caminho_csv, diretorio_artefatos = resolver_caminhos()
    if not caminho_csv.exists():
        print(f"ERRO: CSV não encontrado em {caminho_csv}", file=sys.stderr)
        print(
            "Rode `python scripts/verificar_dataset.py --extrair` antes.",
            file=sys.stderr,
        )
        return 1

    if args.secao == "all":
        secoes_a_rodar = sorted(SECOES.keys())
    else:
        try:
            numero = int(args.secao)
        except ValueError:
            print(f"ERRO: --secao aceita inteiro ou 'all' (recebido: {args.secao!r})", file=sys.stderr)
            return 2
        if numero not in SECOES:
            print(f"ERRO: seção {numero} não implementada. Opções: {sorted(SECOES)}", file=sys.stderr)
            return 2
        secoes_a_rodar = [numero]

    saida_2_cache: dict | None = None
    parada_7_acionada = False
    for numero in secoes_a_rodar:
        print(f"== EDA §{numero} ==")
        if numero == 3 and saida_2_cache is not None:
            saida = _executar_secao_3_a_partir_de(saida_2_cache, diretorio_artefatos)
        elif numero == 8 and parada_7_acionada:
            # Gatilho de parada do plano 002/003: §8 depende de o corpus
            # estar "limpo o bastante" para análise de vocabulário fiel.
            print(
                "§8 pulada: §7 disparou parada (>1% do corpus afetado). "
                "Aguardando política de deduplicação do usuário."
            )
            print()
            continue
        else:
            if numero == 7:
                saida = executar_secao_7(
                    caminho_csv, diretorio_artefatos, n_processos=args.n_processos
                )
            else:
                saida = SECOES[numero](caminho_csv, diretorio_artefatos)
            if numero == 2:
                saida_2_cache = saida
            if numero == 7 and saida.get("parada_acionada"):
                parada_7_acionada = True
        SUMARIOS[numero](saida)
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
