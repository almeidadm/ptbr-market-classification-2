"""
Análise dos artefatos produzidos por `scripts/drift/run_b1.py`.

Funções puras de:
- Descoberta e seleção de "corrida principal" por combo
  (`granularidade`, `escopo`), filtrando smoke tests (corridas com
  `limite_pares` definido nos extras do metadata).
- Agregação estilo Wanderley Tabela 1: mean/std/quantis de p-values
  por (escopo, granularidade, teste, condicao).
- Figura série temporal de p-values (estilo Wanderley Figura 2): uma
  curva por teste para a condição `time_ordered`, com referência
  horizontal em α e banda agregada da condição `randomized`.

Não persiste em disco — a orquestração fica em
`scripts/drift/analise_b1.py`.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.figure as mfigure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import ALPHA_DRIFT, NOMES_TESTES_B1


CORES_TESTES: dict[str, str] = {
    "KS": "#1f77b4",
    "CVM": "#2ca02c",
    "KTS": "#d62728",
    "LSDD": "#9467bd",
}


def descobrir_runs(raiz_b1: Path) -> list[Path]:
    """Lista subdiretórios de `b1_statistical/` que pareçam runs (não auxiliares)."""
    if not raiz_b1.exists():
        return []
    return sorted(
        d
        for d in raiz_b1.iterdir()
        if d.is_dir() and not d.name.startswith("_")
    )


def carregar_run(dir_run: Path) -> tuple[dict, pd.DataFrame] | None:
    """
    Lê (metadata, results) de um diretório de execução. Devolve `None`
    se o run estiver incompleto (falta metadata.json ou results.parquet).
    """
    meta_path = dir_run / "metadata.json"
    parquet_path = dir_run / "results.parquet"
    if not (meta_path.exists() and parquet_path.exists()):
        return None
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    df = pd.read_parquet(parquet_path, engine="pyarrow")
    return metadata, df


def eh_smoke_test(metadata: dict) -> bool:
    """True se o run foi rodado com `--limite-pares` (presente nos extras)."""
    extras = metadata.get("extras") or {}
    return extras.get("limite_pares") is not None


def escolher_corrida_principal_por_combo(
    runs: list[tuple[Path, dict, pd.DataFrame]],
) -> dict[tuple[str, str], tuple[Path, dict, pd.DataFrame]]:
    """
    Para cada (granularidade, escopo), escolhe a corrida principal:
    1. Descarta smoke tests.
    2. Entre os não-smoke, prefere a com mais linhas em `results.parquet`.
    3. Empate → timestamp_iso mais recente.

    Devolve dict `{(granularidade, escopo): (dir, metadata, df)}`.
    """
    agrupados: dict[tuple[str, str], list[tuple[Path, dict, pd.DataFrame]]] = {}
    for dir_run, metadata, df in runs:
        if eh_smoke_test(metadata):
            continue
        chave = (metadata["granularidade"], metadata["escopo"])
        agrupados.setdefault(chave, []).append((dir_run, metadata, df))

    escolhidos: dict[tuple[str, str], tuple[Path, dict, pd.DataFrame]] = {}
    for chave, lista in agrupados.items():
        lista.sort(
            key=lambda t: (len(t[2]), t[1].get("timestamp_iso", "")),
            reverse=True,
        )
        escolhidos[chave] = lista[0]
    return escolhidos


def filtrar_janela_invalida(
    escolhidos: dict[tuple[str, str], tuple[Path, dict, pd.DataFrame]],
    *,
    rotulo: str = "2017-10",
) -> tuple[
    dict[tuple[str, str], tuple[Path, dict, pd.DataFrame]],
    dict[tuple[str, str], int],
]:
    # ADR 011 §D.2 previa excluir out/2017 da grade mensal, mas a janela
    # entrou nas 3 execuções mensais (n_artigos {global=120, mercado=16,
    # nao_mercado=104}). Filtra time_ordered onde a janela aparece em
    # janela_a ou janela_b. Os pares randomized usam rótulos `rand_NNN` —
    # não atingidos por este filtro; baseline randomized ainda contém
    # artigos de out/2017 distribuídos entre as pseudo-janelas. Remoção
    # de origem fica como R2 (re-rodar mensais excluindo a janela).
    filtrados: dict[tuple[str, str], tuple[Path, dict, pd.DataFrame]] = {}
    descartados: dict[tuple[str, str], int] = {}
    for chave, (dir_run, metadata, df) in escolhidos.items():
        if metadata.get("granularidade") != "mensal":
            filtrados[chave] = (dir_run, metadata, df)
            continue
        mascara = (df["janela_a"] == rotulo) | (df["janela_b"] == rotulo)
        n_descartado = int(mascara.sum())
        df_filtrado = df.loc[~mascara].reset_index(drop=True)
        filtrados[chave] = (dir_run, metadata, df_filtrado)
        descartados[chave] = n_descartado
    return filtrados, descartados


def agregar_tabela_wanderley(
    escolhidos: dict[tuple[str, str], tuple[Path, dict, pd.DataFrame]],
) -> pd.DataFrame:
    """
    Para cada (escopo, granularidade, teste, condicao), agrega mean, std
    e quantis 25/50/75 dos p-values. Equivalente ao layout da Tabela 1
    do Wanderley estendido com a coluna `escopo`.
    """
    linhas: list[dict] = []
    for (granularidade, escopo), (_, _, df) in escolhidos.items():
        for teste in NOMES_TESTES_B1:
            for condicao in ("time_ordered", "randomized"):
                serie = df[(df["teste"] == teste) & (df["condicao"] == condicao)][
                    "p_value"
                ]
                if len(serie) == 0:
                    continue
                linhas.append(
                    {
                        "escopo": escopo,
                        "granularidade": granularidade,
                        "teste": teste,
                        "condicao": condicao,
                        "n": int(len(serie)),
                        "mean": float(serie.mean()),
                        "std": float(serie.std(ddof=1)) if len(serie) > 1 else 0.0,
                        "q25": float(serie.quantile(0.25)),
                        "q50": float(serie.quantile(0.50)),
                        "q75": float(serie.quantile(0.75)),
                    }
                )
    return pd.DataFrame(linhas)


def _filtrar_escolhidos(
    escolhidos: dict[tuple[str, str], tuple[Path, dict, pd.DataFrame]],
    *,
    granularidades: set[str],
    escopos: set[str],
) -> dict[tuple[str, str], tuple[Path, dict, pd.DataFrame]]:
    return {
        chave: valor
        for chave, valor in escolhidos.items()
        if chave[0] in granularidades and chave[1] in escopos
    }


def agregar_tabela_replica_wanderley(
    escolhidos: dict[tuple[str, str], tuple[Path, dict, pd.DataFrame]],
) -> pd.DataFrame:
    # Tabela 1 do artigo: réplica direta do Wanderley restrita ao escopo
    # global e à granularidade mensal (primária por ADR 011 §D.2).
    sub = _filtrar_escolhidos(
        escolhidos, granularidades={"mensal"}, escopos={"global"}
    )
    return agregar_tabela_wanderley(sub)


def agregar_tabela_condicional(
    escolhidos: dict[tuple[str, str], tuple[Path, dict, pd.DataFrame]],
) -> pd.DataFrame:
    # Tabela 2 do artigo: contribuição própria — perfil de drift sob
    # análise condicional por classe (mercado × não-mercado), mensal.
    sub = _filtrar_escolhidos(
        escolhidos,
        granularidades={"mensal"},
        escopos={"mercado", "nao_mercado"},
    )
    return agregar_tabela_wanderley(sub)


def agregar_tabela_anexo_bisemanal(
    escolhidos: dict[tuple[str, str], tuple[Path, dict, pd.DataFrame]],
) -> pd.DataFrame:
    # Tabela de anexo: robustez sob a granularidade bi-semanal do
    # Wanderley (ADR 011 §D.2 — "consistência sob granularidade do
    # trabalho de referência").
    sub = _filtrar_escolhidos(
        escolhidos,
        granularidades={"bisemanal"},
        escopos={"global", "mercado", "nao_mercado"},
    )
    return agregar_tabela_wanderley(sub)


def renderizar_tabela_markdown(tabela: pd.DataFrame) -> str:
    """Renderiza a tabela longa em markdown agrupando por (escopo, granularidade)."""
    if tabela.empty:
        return "_(nenhum dado)_\n"
    blocos: list[str] = []
    chaves = (
        tabela[["escopo", "granularidade"]].drop_duplicates().itertuples(index=False)
    )
    for escopo, granularidade in chaves:
        sub = tabela[
            (tabela["escopo"] == escopo) & (tabela["granularidade"] == granularidade)
        ].copy()
        if sub.empty:
            continue
        blocos.append(f"\n### {escopo} — {granularidade}\n")
        blocos.append(
            "| Teste | Condição | n | Mean | STD | Q25 | Q50 | Q75 |\n"
            "|---|---|---:|---:|---:|---:|---:|---:|\n"
        )
        for _, r in sub.iterrows():
            blocos.append(
                f"| {r['teste']} | {r['condicao']} | {r['n']} | "
                f"{r['mean']:.3f} | {r['std']:.3f} | "
                f"{r['q25']:.3f} | {r['q50']:.3f} | {r['q75']:.3f} |\n"
            )
    return "".join(blocos)


def _ordem_x(rotulos: list[str]) -> list[int]:
    """Ordem cronológica dos rótulos. Para temporais funciona via sort lexicográfico."""
    indices = list(range(len(rotulos)))
    indices.sort(key=lambda i: rotulos[i])
    return indices


def _adicionar_par_idx(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adiciona coluna `par_idx` (índice posicional do par dentro de
    cada `(condicao, repeticao, teste)`). Pareia time-ordered com
    randomized pelo par_idx — não pelos rótulos, que são `rand_NNN`
    no caso randomizado.
    """
    df = df.copy()
    df = df.sort_values(["condicao", "repeticao", "teste", "janela_a"], kind="mergesort")
    df["par_idx"] = df.groupby(
        ["condicao", "repeticao", "teste"], sort=False
    ).cumcount()
    return df


def figura_serie_temporal(
    df: pd.DataFrame,
    *,
    escopo: str,
    granularidade: str,
    alpha: float = ALPHA_DRIFT,
    titulo: str | None = None,
) -> mfigure.Figure:
    """
    Plot estilo Wanderley Figura 2: p-value de cada par-de-janelas-
    consecutivas para a condição `time_ordered`, uma linha por teste.

    Banda cinza ao fundo mostra o envelope min-max dos baselines
    randomizados (5 repetições) para o teste KTS — útil pra evidenciar
    visualmente o gap entre time-ordered e o nulo. O pareamento é por
    `par_idx` (posicional), porque os rótulos `rand_NNN` não coincidem
    com os temporais.
    """
    df = _adicionar_par_idx(df)
    df_time = df[df["condicao"] == "time_ordered"]
    if df_time.empty:
        raise ValueError(f"Sem dados time_ordered para {escopo}/{granularidade}.")

    rotulos_por_idx = (
        df_time[df_time["teste"] == NOMES_TESTES_B1[0]]
        .sort_values("par_idx")[["par_idx", "janela_b"]]
        .drop_duplicates()
        .set_index("par_idx")["janela_b"]
    )

    figura, eixo = plt.subplots(figsize=(10, 4))

    # Envelope randomized (KTS) como referência do nulo
    df_rand_kts = df[(df["condicao"] == "randomized") & (df["teste"] == "KTS")]
    if not df_rand_kts.empty:
        rand_por_par = df_rand_kts.groupby("par_idx")["p_value"].agg(["min", "max"])
        eixo.fill_between(
            rand_por_par.index,
            rand_por_par["min"],
            rand_por_par["max"],
            color="#cccccc",
            alpha=0.4,
            label="KTS randomized (min-max)",
        )

    for teste in NOMES_TESTES_B1:
        sub = df_time[df_time["teste"] == teste].sort_values("par_idx")
        if sub.empty:
            continue
        eixo.plot(
            sub["par_idx"],
            sub["p_value"],
            marker="o",
            markersize=3,
            linewidth=1,
            color=CORES_TESTES.get(teste, "black"),
            label=teste,
        )

    eixo.axhline(
        alpha,
        color="red",
        linestyle="--",
        linewidth=0.8,
        label=f"α = {alpha}",
    )

    eixo.set_xticks(rotulos_por_idx.index)
    eixo.set_xticklabels(rotulos_por_idx.values, rotation=60, ha="right", fontsize=7)
    eixo.set_ylim(-0.02, 1.02)
    eixo.set_ylabel("p-value")
    eixo.set_xlabel("Janela B (vs. anterior)")
    eixo.set_title(
        titulo or f"p-value time-ordered — {escopo} / {granularidade}"
    )
    eixo.legend(loc="upper right", fontsize=8, ncol=2)
    eixo.grid(True, linestyle=":", alpha=0.5)
    figura.tight_layout()
    return figura


def figura_panel_escopos(
    escolhidos: dict[tuple[str, str], tuple[Path, dict, pd.DataFrame]],
    *,
    granularidade: str,
    alpha: float = ALPHA_DRIFT,
) -> mfigure.Figure:
    """
    Painel side-by-side dos três escopos (global, mercado, nao_mercado)
    para uma granularidade fixa. Mesma escala y para comparação visual.
    """
    escopos = ("global", "mercado", "nao_mercado")
    figura, eixos = plt.subplots(
        nrows=3, ncols=1, figsize=(10, 9), sharex=True, sharey=True
    )

    for eixo, escopo in zip(eixos, escopos):
        chave = (granularidade, escopo)
        if chave not in escolhidos:
            eixo.set_title(f"{escopo}: sem dados")
            continue
        _, _, df = escolhidos[chave]
        df = _adicionar_par_idx(df)
        df_time = df[df["condicao"] == "time_ordered"]
        rotulos_por_idx = (
            df_time[df_time["teste"] == NOMES_TESTES_B1[0]]
            .sort_values("par_idx")[["par_idx", "janela_b"]]
            .drop_duplicates()
            .set_index("par_idx")["janela_b"]
        )
        for teste in NOMES_TESTES_B1:
            sub = df_time[df_time["teste"] == teste].sort_values("par_idx")
            if sub.empty:
                continue
            eixo.plot(
                sub["par_idx"],
                sub["p_value"],
                marker="o",
                markersize=3,
                linewidth=1,
                color=CORES_TESTES.get(teste, "black"),
                label=teste,
            )
        eixo.axhline(alpha, color="red", linestyle="--", linewidth=0.8)
        eixo.set_ylabel("p-value")
        eixo.set_title(escopo)
        eixo.set_ylim(-0.02, 1.02)
        eixo.grid(True, linestyle=":", alpha=0.5)
        eixo.set_xticks(rotulos_por_idx.index)
        eixo.set_xticklabels(
            rotulos_por_idx.values, rotation=60, ha="right", fontsize=6
        )

    eixos[0].legend(loc="upper right", fontsize=8, ncol=4)
    figura.suptitle(f"p-value time-ordered por escopo — {granularidade}", y=1.00)
    figura.tight_layout()
    return figura


def resumir_achados(tabela: pd.DataFrame) -> str:
    """Texto curto com os principais achados a partir da tabela agregada."""
    if tabela.empty:
        return "_(nenhum dado)_\n"

    linhas: list[str] = []
    linhas.append("## Achados resumidos\n")
    linhas.append(
        f"\n_Gerado em {datetime.now(timezone.utc).isoformat()}._\n\n"
    )

    # Pivot: (escopo, granularidade) × condicao → mean p_value (média sobre testes)
    pivot = (
        tabela.groupby(["escopo", "granularidade", "condicao"])["mean"]
        .mean()
        .unstack("condicao")
    )
    linhas.append(
        "**p-value médio (média sobre os 4 testes) por escopo e granularidade:**\n\n"
    )
    linhas.append("| Escopo | Granularidade | time-ordered | randomized | gap |\n")
    linhas.append("|---|---|---:|---:|---:|\n")
    for (escopo, granularidade), row in pivot.iterrows():
        to = row.get("time_ordered", float("nan"))
        ra = row.get("randomized", float("nan"))
        gap = ra - to if not (np.isnan(to) or np.isnan(ra)) else float("nan")
        linhas.append(
            f"| {escopo} | {granularidade} | {to:.3f} | {ra:.3f} | {gap:+.3f} |\n"
        )

    return "".join(linhas)
