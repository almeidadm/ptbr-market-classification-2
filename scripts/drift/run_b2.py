"""
Executa o bloco B2 (análise semântica) da detecção de drift (ADRs 010
e 011) para um par (`granularidade`, `escopo`).

Mesmo pipeline de carregamento e janelamento do B1, trocando o conjunto
de testes pelas três métricas semânticas (`cosseno_centroide_consecutivo`,
`cosseno_centroide_cumulativo`, `mmd2_consecutivo`).

Persiste em `artifacts/drift/b2_semantic/<timestamp>-<granularidade>-<escopo>/`
seguindo o contrato da ADR 011 §D.6 (metadata.json + results.parquet com
colunas `janela_a, janela_b, metrica, repeticao, condicao, valor`).

Custo:

- Cosseno (consecutivo + cumulativo) é praticamente grátis.
- MMD² domina via `MMDDrift(n_permutations=1)`. Estimativa: ~10-20 s por
  par em CPU; ~5-15 min por combo mensal (32 pares × 6 condições).
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

RAIZ_PROJETO = Path(__file__).resolve().parent.parent.parent
if str(RAIZ_PROJETO) not in sys.path:
    sys.path.insert(0, str(RAIZ_PROJETO))

import numpy as np
import pandas as pd

from src.config import (
    DATA_FIM_DRIFT_EXCLUSIVO,
    DATA_INICIO_DRIFT,
    DIR_ARTEFATOS_DRIFT,
    DIR_DADOS_PROCESSADO,
    ESCOPOS_DRIFT,
    GRANULARIDADES_DRIFT,
    N_REPETICOES_DRIFT,
    SEED,
)
from src.drift.artefatos import (
    construir_metadata_drift,
    montar_dir_drift,
    salvar_json,
    salvar_resultados,
)
from src.drift.io_drift import (
    PADRAO_CORPUS,
    PADRAO_EMBEDDINGS,
    carregar_e_alinhar,
    empilhar_embeddings,
    filtrar_por_escopo,
    gerar_janelas,
)
from src.drift.janelas import (
    filtrar_periodo_efetivo,
    gerar_repeticoes_aleatorizadas,
)
from src.drift.semantica import NOMES_METRICAS_B2, aplicar_todas


def executar_condicao(
    embeddings_por_janela: list[np.ndarray],
    rotulos: list[str],
    *,
    condicao: str,
    repeticao_id: int,
    seed: int,
    device: str | None,
    log_prefixo: str = "",
) -> list[dict]:
    """
    Aplica as três métricas no conjunto de janelas e devolve linhas
    prontas para virar DataFrame.
    """
    t0 = time.perf_counter()
    resultados = aplicar_todas(
        embeddings_por_janela, rotulos, seed=seed, device=device
    )
    decorrido = time.perf_counter() - t0
    print(
        f"  {log_prefixo}{len(resultados)} medições "
        f"em {decorrido / 60:.1f} min",
        flush=True,
    )
    return [
        {
            "janela_a": r.janela_a,
            "janela_b": r.janela_b,
            "metrica": r.metrica,
            "repeticao": repeticao_id,
            "condicao": condicao,
            "valor": r.valor,
        }
        for r in resultados
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--embeddings",
        type=Path,
        default=Path(
            os.environ.get("PTBR_MC_EMBEDDINGS_DRIFT") or PADRAO_EMBEDDINGS
        ),
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        default=Path(os.environ.get("PTBR_MC_CORPUS_DRIFT") or PADRAO_CORPUS),
    )
    parser.add_argument("--out", type=Path, default=DIR_ARTEFATOS_DRIFT)
    parser.add_argument(
        "--granularidade", choices=list(GRANULARIDADES_DRIFT), required=True
    )
    parser.add_argument("--escopo", choices=list(ESCOPOS_DRIFT), required=True)
    parser.add_argument(
        "--n-repeticoes", type=int, default=N_REPETICOES_DRIFT
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda", "gpu"],
        default=None,
        help="Dispositivo para MMD² (default: auto-detect).",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    print(f"[1/5] Carregando embeddings e corpus...")
    df = carregar_e_alinhar(args.embeddings, args.corpus)
    print(f"       {len(df):,} artigos alinhados.")

    n_antes_periodo = len(df)
    df = filtrar_periodo_efetivo(df)
    print(
        f"       cobertura efetiva ADR 003 "
        f"[{DATA_INICIO_DRIFT}, {DATA_FIM_DRIFT_EXCLUSIVO}): "
        f"{len(df):,} artigos "
        f"(descartados {n_antes_periodo - len(df):,})."
    )

    print(f"[2/5] Filtrando por escopo={args.escopo}...")
    df_filtrado = filtrar_por_escopo(df, args.escopo)
    print(f"       {len(df_filtrado):,} artigos após filtragem.")

    print(f"[3/5] Gerando janelas {args.granularidade}...")
    janelas = gerar_janelas(df_filtrado, args.granularidade)
    print(f"       {len(janelas)} janelas geradas.")
    if len(janelas) < 2:
        print("ERRO: menos de 2 janelas — nada para comparar.", file=sys.stderr)
        return 2

    print(f"[3/5] Pré-empilhando embeddings por janela...")
    emb_temporais = [
        empilhar_embeddings(df_filtrado, j.indices_no_corpus) for j in janelas
    ]
    rotulos_temporais = [j.rotulo for j in janelas]

    destino = montar_dir_drift(
        "b2_semantic",
        granularidade=args.granularidade,
        escopo=args.escopo,
        raiz=args.out,
    )
    caminho_parquet = destino / "results.parquet"
    if caminho_parquet.exists() and not args.force:
        print(
            f"ERRO: {caminho_parquet} já existe. Use --force para sobrescrever.",
            file=sys.stderr,
        )
        return 3

    t_inicio = time.perf_counter()

    print(f"[4/5] Time-ordered: {len(janelas) - 1} pares...")
    linhas = executar_condicao(
        emb_temporais,
        rotulos_temporais,
        condicao="time_ordered",
        repeticao_id=0,
        seed=SEED,
        device=args.device,
        log_prefixo="[time-ordered] ",
    )

    print(
        f"[5/5] Randomized: {args.n_repeticoes} repetições × "
        f"{len(janelas) - 1} pares..."
    )
    repeticoes = gerar_repeticoes_aleatorizadas(
        n_total=len(df_filtrado),
        janelas_referencia=janelas,
        seed_global=SEED,
        n_repeticoes=args.n_repeticoes,
    )
    for rep_id, janelas_rand in enumerate(repeticoes):
        print(f"  repetição {rep_id + 1}/{args.n_repeticoes}")
        emb_rand = [
            empilhar_embeddings(df_filtrado, j.indices_no_corpus)
            for j in janelas_rand
        ]
        rotulos_rand = [j.rotulo for j in janelas_rand]
        linhas_rep = executar_condicao(
            emb_rand,
            rotulos_rand,
            condicao="randomized",
            repeticao_id=rep_id,
            seed=SEED + rep_id + 1,
            device=args.device,
            log_prefixo=f"[rand rep={rep_id}] ",
        )
        linhas.extend(linhas_rep)

    duracao = time.perf_counter() - t_inicio
    print(f"\nGravando resultados em {destino}...")
    df_resultados = pd.DataFrame(linhas)
    salvar_resultados(destino, df_resultados)

    metadata = construir_metadata_drift(
        bloco="b2_semantic",
        escopo=args.escopo,
        granularidade=args.granularidade,
        corpus=args.corpus,
        n_artigos=len(df_filtrado),
        embedding="bertimbau_base_cls",
        embedding_path=args.embeddings,
        janelas=[j.resumir() for j in janelas],
        tests=NOMES_METRICAS_B2,
        duracao_segundos=duracao,
        extras={
            "device_solicitado": args.device,
            "n_pares_time_ordered": len(janelas) - 1,
            "n_pares_randomized_total": args.n_repeticoes * (len(janelas) - 1),
            "mmd2_n_permutations": 1,
            "mmd2_kernel": "rbf_mediana",
        },
    )
    salvar_json(destino / "metadata.json", metadata)

    tamanho_kb = caminho_parquet.stat().st_size / 1024
    print(
        f"  results.parquet ({tamanho_kb:.1f} KB, {len(df_resultados)} linhas) "
        f"+ metadata.json"
    )
    print(f"  duração total: {duracao / 60:.1f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
