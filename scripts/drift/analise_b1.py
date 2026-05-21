"""
Análise dos artefatos de B1 (testes estatísticos de drift, ADR 011).

Lê todos os diretórios `artifacts/drift/b1_statistical/<run>/`, escolhe
a corrida principal por (granularidade, escopo) descartando smoke
tests, e produz em `artifacts/drift/analises/b1/<timestamp>/`:

- `tabela_wanderley.csv` e `.md`: agregação estilo Tabela 1 do Wanderley
  por (escopo, granularidade, teste, condicao).
- `figura_serie_temporal_<escopo>_<granularidade>.{png,svg}`: p-value
  time-ordered por par-de-janelas consecutivas, uma curva por teste,
  linha em α e banda min-max do randomized.
- `figura_painel_<granularidade>.{png,svg}`: três escopos empilhados.
- `analise.md`: resumo textual dos achados.
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

from src.config import DIR_ARTEFATOS_DRIFT, GRANULARIDADES_DRIFT
from src.drift.analise_b1 import (
    agregar_tabela_wanderley,
    carregar_run,
    descobrir_runs,
    escolher_corrida_principal_por_combo,
    figura_panel_escopos,
    figura_serie_temporal,
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

    print(f"[2/5] Selecionando corrida principal por combo...")
    escolhidos = escolher_corrida_principal_por_combo(runs)
    if not escolhidos:
        print(
            "ERRO: nenhuma corrida não-smoke encontrada. Rode run_b1.py "
            "sem --limite-pares antes.",
            file=sys.stderr,
        )
        return 3
    for (granularidade, escopo), (d, m, df) in sorted(escolhidos.items()):
        print(f"  {granularidade}/{escopo} → {d.name} ({len(df)} linhas)")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M")
    destino = args.out / timestamp
    destino.mkdir(parents=True, exist_ok=True)
    print(f"[3/5] Gravando em {destino}")

    print(f"[4/5] Agregando tabela Wanderley...")
    tabela = agregar_tabela_wanderley(escolhidos)
    tabela.to_csv(destino / "tabela_wanderley.csv", index=False)
    (destino / "tabela_wanderley.md").write_text(
        "# Tabela Wanderley (B1)\n" + renderizar_tabela_markdown(tabela),
        encoding="utf-8",
    )

    print(f"[5/5] Gerando figuras...")
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

    analise_md = (
        f"# Análise B1 — {timestamp}\n\n"
        + resumir_achados(tabela)
        + "\n\n## Corridas selecionadas\n\n"
        + "| Combo | Diretório | Linhas | Duração (min) | Commit |\n"
        + "|---|---|---:|---:|---|\n"
        + "".join(
            f"| {r['combo']} | {r['diretorio']} | {r['n_linhas']} | "
            f"{r['duracao_minutos']} | "
            f"`{r['git_commit'][:7] if r['git_commit'] else 'n/a'}` |\n"
            for r in runs_selecionados
        )
        + "\n## Tabela Wanderley\n\n"
        + renderizar_tabela_markdown(tabela)
    )
    (destino / "analise.md").write_text(analise_md, encoding="utf-8")
    print(f"\nAnálise gravada em {destino}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
