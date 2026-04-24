"""
Análise complementar à EDA consolidada (§1-§7), pedida para informar a
decisão de agrupamento de categorias (potencial 004).

Reutiliza as funções já validadas em `src/eda/` — não introduz lógica nova.

Produz:
  - tabelas/06b-jaccard-mercado-vs-todas-categorias.csv
      Jaccard top-30 entre `mercado` e cada uma das 47 outras categorias,
      mais colunas de `n_amostras` e `n_tokens_extraidos` para contexto.
  - tabelas/03b-deriva-por-categoria.csv
      Mann-Kendall, z-proporções (primeiro × último ano disponível),
      Qui² k×2 sobre a prevalência de cada categoria (one-vs-rest) na
      série mensal e nos blocos anuais.

Uso:
    python scripts/analise_categorias_complementar.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pandas as pd

RAIZ_PROJETO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ_PROJETO))

from src.eda.carregamento import carregar_articles
from src.eda.contagens import serie_total_e_classe
from src.eda.testes_temporais import (
    mann_kendall,
    qui_quadrado_k_por_2,
    z_proporcoes,
)
from src.eda.vocabulario import sobreposicao_top_tokens, tokens_top_k_tfidf

SEED = 20260423
DATA_LIMITE_MENSAL = pd.Timestamp("2017-10-01")


def resolver_caminhos() -> tuple[Path, Path]:
    csv = Path(os.environ.get("CAMINHO_CSV", RAIZ_PROJETO / "data" / "raw" / "articles.csv"))
    artefatos = Path(
        os.environ.get("DIRETORIO_ARTEFATOS", RAIZ_PROJETO / "artifacts" / "eda")
    )
    return csv, artefatos


def jaccard_mercado_vs_todas(df: pd.DataFrame, dir_tabelas: Path) -> pd.DataFrame:
    """Jaccard top-30 entre mercado e as 47 demais categorias.

    Categorias com textos suficientes para passar `min_df=2` no TF-IDF
    aparecem com top-30 preenchido; as muito pequenas aparecem truncadas
    (menos de 30 tokens extraídos) — o campo `n_tokens_extraidos` ajuda a
    filtrar resultados não confiáveis.
    """
    contagem = df["category"].value_counts(dropna=True)
    categorias_todas = [str(c) for c in contagem.index]
    if "mercado" not in categorias_todas:
        raise RuntimeError("categoria 'mercado' ausente do corpus")

    # Pular categorias com < 2 textos não-nulos: TF-IDF com min_df=2
    # filtra todos os tokens e não produz saída útil.
    com_texto = df.loc[df["text"].notna(), "category"].value_counts()
    categorias_viaveis = [
        c for c in categorias_todas if int(com_texto.get(c, 0)) >= 2
    ]
    descartadas = [c for c in categorias_todas if c not in categorias_viaveis]
    if descartadas:
        print(
            f"[jaccard] descartando {len(descartadas)} categorias com <2 textos: {descartadas}",
            flush=True,
        )

    print(
        f"[jaccard] calculando top-30 TF-IDF para {len(categorias_viaveis)} categorias...",
        flush=True,
    )
    top_tokens = tokens_top_k_tfidf(
        df=df,
        coluna_categoria="category",
        coluna_texto="text",
        categorias=categorias_viaveis,
        k=30,
        n_amostra_por_categoria=2000,
        seed=SEED,
    )

    sobreposicao = sobreposicao_top_tokens(top_tokens, categoria_referencia="mercado")

    # Enriquecer com contagem de amostras e quantos tokens foram efetivamente
    # extraídos (útil para confiabilidade do Jaccard em categorias pequenas).
    tokens_por_cat = (
        top_tokens.groupby("categoria").size().rename("n_tokens_extraidos")
    )
    sobreposicao = sobreposicao.merge(
        tokens_por_cat.rename("n_tokens_extraidos"),
        how="left",
        left_on="categoria_vs",
        right_index=True,
    )
    sobreposicao["n_textos_categoria"] = sobreposicao["categoria_vs"].map(
        com_texto.to_dict()
    )
    sobreposicao["n_total_categoria"] = sobreposicao["categoria_vs"].map(
        contagem.to_dict()
    )

    # Ordem final: Jaccard desc; categorias descartadas entram com NaN no final.
    descartadas_df = pd.DataFrame(
        [
            {
                "categoria_vs": c,
                "jaccard_top30": None,
                "tokens_compartilhados": "",
                "tokens_exclusivos_referencia": "",
                "n_tokens_extraidos": 0,
                "n_textos_categoria": int(com_texto.get(c, 0)),
                "n_total_categoria": int(contagem.get(c, 0)),
            }
            for c in descartadas
        ]
    )
    resultado = pd.concat([sobreposicao, descartadas_df], axis=0, ignore_index=True)

    caminho = dir_tabelas / "06b-jaccard-mercado-vs-todas-categorias.csv"
    resultado.to_csv(caminho, index=False, encoding="utf-8")
    print(f"[jaccard] persistido em {caminho}", flush=True)
    return resultado


def _testes_para_categoria(df: pd.DataFrame, categoria: str) -> dict:
    """Aplica os 3 testes de §3 à categoria dada (formulação one-vs-rest)."""
    alvo = f"_eh_{categoria.replace('-', '_')}"
    df = df.assign(**{alvo: (df["category"] == categoria)})

    sub_mensal = df.loc[df["date"] < DATA_LIMITE_MENSAL]
    serie = serie_total_e_classe(sub_mensal, coluna_alvo_bool=alvo, granularidade="M")

    mk = mann_kendall(serie["prevalencia"].to_numpy())

    # Tabela anual: total e positivos por ano-calendário.
    anual = (
        df.assign(ano=df["date"].dt.year)
        .groupby("ano", sort=True)
        .agg(total=("category", "size"), positivos=(alvo, "sum"))
        .reset_index()
    )
    anual["negativos"] = anual["total"] - anual["positivos"]
    anual = anual.loc[anual["total"] > 0].reset_index(drop=True)

    zp_efeito = None
    zp_p = None
    zp_estat = None
    if len(anual) >= 2:
        primeiro = anual.iloc[0]
        ultimo = anual.iloc[-1]
        try:
            zp = z_proporcoes(
                n1=int(primeiro["total"]),
                k1=int(primeiro["positivos"]),
                n2=int(ultimo["total"]),
                k2=int(ultimo["positivos"]),
            )
            zp_estat = zp.estatistica
            zp_p = zp.p_valor
            zp_efeito = {
                **zp.efeito,
                "ano_1": int(primeiro["ano"]),
                "ano_2": int(ultimo["ano"]),
            }
        except ValueError:
            pass

    qui_estat = None
    qui_p = None
    qui_efeito = None
    if len(anual) >= 2 and anual["positivos"].sum() > 0:
        try:
            tabela = anual[["positivos", "negativos"]].to_numpy()
            qui = qui_quadrado_k_por_2(tabela)
            qui_estat = qui.estatistica
            qui_p = qui.p_valor
            qui_efeito = {**qui.efeito, "anos": anual["ano"].tolist()}
        except ValueError:
            pass

    return {
        "categoria": categoria,
        "n_total": int((df["category"] == categoria).sum()),
        "n_meses_cobertos": int((serie["positivos"] > 0).sum()),
        "prevalencia_primeiro_mes": float(serie["prevalencia"].iloc[0])
        if len(serie) > 0
        else None,
        "prevalencia_ultimo_mes": float(serie["prevalencia"].iloc[-1])
        if len(serie) > 0
        else None,
        "mk_tau": mk.estatistica,
        "mk_p": mk.p_valor,
        "mk_efeito": json.dumps(mk.efeito, ensure_ascii=False),
        "zp_estatistica": zp_estat,
        "zp_p": zp_p,
        "zp_efeito": json.dumps(zp_efeito, ensure_ascii=False)
        if zp_efeito is not None
        else None,
        "qui_estatistica": qui_estat,
        "qui_p": qui_p,
        "qui_efeito": json.dumps(qui_efeito, ensure_ascii=False)
        if qui_efeito is not None
        else None,
    }


def deriva_por_categoria(df: pd.DataFrame, dir_tabelas: Path) -> pd.DataFrame:
    """Aplica §3 (3 testes) a cada categoria one-vs-rest."""
    categorias = sorted(df["category"].dropna().unique().tolist())
    print(
        f"[deriva] rodando 3 testes para cada uma das {len(categorias)} categorias...",
        flush=True,
    )

    linhas = [_testes_para_categoria(df, c) for c in categorias]
    resultado = pd.DataFrame(linhas).sort_values(
        "n_total", ascending=False, kind="mergesort"
    )

    caminho = dir_tabelas / "03b-deriva-por-categoria.csv"
    resultado.to_csv(caminho, index=False, encoding="utf-8")
    print(f"[deriva] persistido em {caminho}", flush=True)
    return resultado


def main() -> int:
    caminho_csv, diretorio_artefatos = resolver_caminhos()
    if not caminho_csv.exists():
        print(f"ERRO: CSV não encontrado em {caminho_csv}", file=sys.stderr)
        return 1

    dir_tabelas = diretorio_artefatos / "tabelas"
    dir_tabelas.mkdir(parents=True, exist_ok=True)

    print(f"[carregamento] lendo {caminho_csv}...", flush=True)
    df = carregar_articles(caminho_csv)
    print(f"[carregamento] {len(df):,} linhas.".replace(",", "."), flush=True)

    jaccard = jaccard_mercado_vs_todas(df, dir_tabelas)
    deriva = deriva_por_categoria(df, dir_tabelas)

    print()
    print("=== Top-15 categorias por Jaccard com mercado (excluindo top-7 já conhecido) ===")
    top7 = {"poder", "colunas", "mercado", "esporte", "mundo", "cotidiano", "ilustrada"}
    cauda_jacc = jaccard.loc[~jaccard["categoria_vs"].isin(top7)].copy()
    print(
        cauda_jacc.head(15)[
            ["categoria_vs", "jaccard_top30", "n_textos_categoria", "n_tokens_extraidos"]
        ].to_string(index=False)
    )

    print()
    print("=== Top-10 categorias com deriva positiva mais forte (Mann-Kendall τ) ===")
    positivas = deriva.sort_values("mk_tau", ascending=False, na_position="last")
    print(
        positivas.head(10)[
            ["categoria", "n_total", "mk_tau", "mk_p", "prevalencia_primeiro_mes", "prevalencia_ultimo_mes"]
        ].to_string(index=False)
    )

    print()
    print("=== Top-10 categorias com deriva negativa mais forte ===")
    negativas = deriva.sort_values("mk_tau", ascending=True, na_position="last")
    print(
        negativas.head(10)[
            ["categoria", "n_total", "mk_tau", "mk_p", "prevalencia_primeiro_mes", "prevalencia_ultimo_mes"]
        ].to_string(index=False)
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
