"""
Análise dos artefatos de B1 (testes estatísticos de drift, ADR 011).

Lê todos os diretórios `artifacts/drift/b1_statistical/<run>/`, escolhe
a corrida principal por (granularidade, escopo) descartando smoke
tests, aplica o filtro R1 (descarta pares envolvendo a janela mensal
`2017-10`, fora do escopo previsto na ADR 011 §D.2) e produz em
`artifacts/drift/analises/b1/<timestamp>/`:

- `tabela1_replica_wanderley.csv` e `.md`: réplica da Tabela 1 do
  Wanderley restrita a escopo global e granularidade mensal.
- `tabela2_condicional.csv` e `.md`: análise condicional por classe
  (mercado × não-mercado) na granularidade mensal.
- `tabela_anexo_bisemanal.csv` e `.md`: tabela de robustez sob a
  granularidade bi-semanal do trabalho de referência (anexo).
- `figura_serie_temporal_<escopo>_<granularidade>.{png,svg}`: p-value
  time-ordered por par-de-janelas consecutivas, uma curva por teste,
  linha em α e banda min-max do randomized.
- `figura_painel_<granularidade>.{png,svg}`: três escopos empilhados.
- `analise.md`: resumo textual dos achados, com detalhamento do R1.
- `runs_selecionados.json`: trilha de auditoria das corridas escolhidas.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

RAIZ_PROJETO = Path(__file__).resolve().parent.parent.parent
if str(RAIZ_PROJETO) not in sys.path:
    sys.path.insert(0, str(RAIZ_PROJETO))

import matplotlib

matplotlib.use("Agg")  # backend headless — não exige display
import matplotlib.pyplot as plt
import pandas as pd

from src.config import DIR_ARTEFATOS_DRIFT, GRANULARIDADES_DRIFT
from src.drift.analise_b1 import (
    agregar_tabela_anexo_bisemanal,
    agregar_tabela_condicional,
    agregar_tabela_replica_wanderley,
    carregar_run,
    descobrir_runs,
    escolher_corrida_principal_por_combo,
    figura_panel_escopos,
    figura_serie_temporal,
    filtrar_janela_invalida,
    renderizar_tabela_markdown,
    resumir_achados,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raiz-b1",
        type=Path,
        default=Path(os.environ.get("PTBR_MC_DIR_DRIFT", DIR_ARTEFATOS_DRIFT))
        / "b1_statistical",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(os.environ.get("PTBR_MC_DIR_DRIFT", DIR_ARTEFATOS_DRIFT))
        / "analises"
        / "b1",
    )
    args = parser.parse_args(argv)

    print(f"[1/5] Descobrindo runs em {args.raiz_b1}...")
    dirs = descobrir_runs(args.raiz_b1)
    if not dirs:
        print(f"ERRO: nenhum run encontrado em {args.raiz_b1}", file=sys.stderr)
        return 2

    runs = []
    for d in dirs:
        carregado = carregar_run(d)
        if carregado is None:
            print(f"  ignorando {d.name}: faltam metadata.json ou results.parquet")
            continue
        metadata, df = carregado
        runs.append((d, metadata, df))
        print(f"  {d.name}: {len(df)} linhas")

    print(f"[2/6] Selecionando corrida principal por combo...")
    escolhidos = escolher_corrida_principal_por_combo(runs)
    if not escolhidos:
        print(
            "ERRO: nenhuma corrida não-smoke encontrada. Rode run_b1.py "
            "sem --limite-pares antes.",
            file=sys.stderr,
        )
        return 3
    for (granularidade, escopo), (d, _, df) in sorted(escolhidos.items()):
        print(f"  {granularidade}/{escopo} → {d.name} ({len(df)} linhas)")

    print(f"[3/6] Aplicando R1 (filtrar janela 2017-10 nas mensais)...")
    escolhidos, descartados_r1 = filtrar_janela_invalida(escolhidos)
    for chave, n in sorted(descartados_r1.items()):
        granularidade, escopo = chave
        print(f"  {granularidade}/{escopo}: {n} linhas descartadas (R1)")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M")
    destino = args.out / timestamp
    destino.mkdir(parents=True, exist_ok=True)
    print(f"[4/6] Gravando em {destino}")

    print(f"[5/6] Agregando tabelas separadas (réplica + condicional + anexo)...")
    tabela_replica = agregar_tabela_replica_wanderley(escolhidos)
    tabela_condicional = agregar_tabela_condicional(escolhidos)
    tabela_anexo = agregar_tabela_anexo_bisemanal(escolhidos)

    for nome, tabela, titulo in [
        ("tabela1_replica_wanderley", tabela_replica, "Tabela 1 — Réplica Wanderley (global, mensal)"),
        ("tabela2_condicional", tabela_condicional, "Tabela 2 — Análise condicional (mercado × não-mercado, mensal)"),
        ("tabela_anexo_bisemanal", tabela_anexo, "Tabela de anexo — Robustez sob granularidade bi-semanal"),
    ]:
        tabela.to_csv(destino / f"{nome}.csv", index=False)
        (destino / f"{nome}.md").write_text(
            f"# {titulo}\n" + renderizar_tabela_markdown(tabela),
            encoding="utf-8",
        )

    print(f"[6/6] Gerando figuras...")
    for (granularidade, escopo), (_, _, df) in escolhidos.items():
        fig = figura_serie_temporal(df, escopo=escopo, granularidade=granularidade)
        base = destino / f"figura_serie_temporal_{escopo}_{granularidade}"
        fig.savefig(f"{base}.png", dpi=150, bbox_inches="tight")
        fig.savefig(f"{base}.svg", bbox_inches="tight")
        plt.close(fig)
        print(f"  {base.name}.{{png,svg}}")

    for granularidade in GRANULARIDADES_DRIFT:
        chaves_disponiveis = [
            (g, e) for (g, e) in escolhidos.keys() if g == granularidade
        ]
        if not chaves_disponiveis:
            continue
        fig = figura_panel_escopos(escolhidos, granularidade=granularidade)
        base = destino / f"figura_painel_{granularidade}"
        fig.savefig(f"{base}.png", dpi=150, bbox_inches="tight")
        fig.savefig(f"{base}.svg", bbox_inches="tight")
        plt.close(fig)
        print(f"  {base.name}.{{png,svg}}")

    runs_selecionados = [
        {
            "combo": f"{granularidade}/{escopo}",
            "diretorio": d.name,
            "n_linhas": int(len(df)),
            "timestamp_iso": m.get("timestamp_iso"),
            "git_commit": m.get("git_commit"),
            "duracao_minutos": (
                round(m.get("duracao_segundos", 0) / 60, 2)
                if m.get("duracao_segundos")
                else None
            ),
        }
        for (granularidade, escopo), (d, m, df) in sorted(escolhidos.items())
    ]
    (destino / "runs_selecionados.json").write_text(
        json.dumps(runs_selecionados, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    tabela_concatenada = pd.concat(
        [tabela_replica, tabela_condicional, tabela_anexo], ignore_index=True
    )

    linhas_r1 = "".join(
        f"| {chave[0]}/{chave[1]} | {n} |\n"
        for chave, n in sorted(descartados_r1.items())
    ) or "| _(nada)_ | 0 |\n"

    analise_md = (
        f"# Análise B1 — {timestamp}\n\n"
        + "## Filtro R1 (out/2017)\n\n"
        + "Pares descartados onde `janela_a` ou `janela_b == '2017-10'`.\n"
        + "Afeta apenas execuções mensais e somente `time_ordered`; "
        + "baseline `randomized` ainda contém artigos da janela inválida "
        + "(remoção de origem fica como pendência R2).\n\n"
        + "| Combo | Linhas descartadas |\n|---|---:|\n"
        + linhas_r1
        + "\n"
        + resumir_achados(tabela_concatenada)
        + "\n\n## Corridas selecionadas\n\n"
        + "| Combo | Diretório | Linhas | Duração (min) | Commit |\n"
        + "|---|---|---:|---:|---|\n"
        + "".join(
            f"| {r['combo']} | {r['diretorio']} | {r['n_linhas']} | "
            f"{r['duracao_minutos']} | "
            f"`{r['git_commit'][:7] if r['git_commit'] else 'n/a'}` |\n"
            for r in runs_selecionados
        )
        + "\n## Tabela 1 — Réplica Wanderley (global, mensal)\n\n"
        + renderizar_tabela_markdown(tabela_replica)
        + "\n## Tabela 2 — Análise condicional (mercado × não-mercado, mensal)\n\n"
        + renderizar_tabela_markdown(tabela_condicional)
        + "\n## Tabela de anexo — Robustez sob granularidade bi-semanal\n\n"
        + renderizar_tabela_markdown(tabela_anexo)
    )
    (destino / "analise.md").write_text(analise_md, encoding="utf-8")
    print(f"\nAnálise gravada em {destino}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
