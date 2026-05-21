"""
Executa o bloco B1 (testes estatísticos) da detecção de drift (ADRs 010
e 011) para um par (`granularidade`, `escopo`).

Carrega:
- `artifacts/drift/embeddings/bertimbau_base_cls/embeddings.parquet`
  (gerado por `scripts/drift/compute_embeddings.py`).
- `data/processado/corpus_opcao7.parquet` para `y_original`, usada na
  filtragem por escopo (`mercado`, `nao_mercado`).

Para cada par de janelas consecutivas:
1. Aplica KS, CVM, KTS e LSDD via `src.drift.testes_estatisticos`.
2. Repete para 5 partições aleatorizadas independentes (seeds
   `SEED + repeticao_id`).

Persiste em
`artifacts/drift/b1_statistical/<timestamp>-<granularidade>-<escopo>/`
seguindo o contrato da ADR 011 §D.6 (metadata.json + results.parquet
com colunas `janela_a, janela_b, teste, repeticao, condicao, p_value,
estatistica`).

KTS e LSDD dominam o custo (O(N²·d) por par). Em L4 GPU com `n_permutations=100`,
estimado ~1–2 min por par-de-janelas-mensal-global; ~30–60 min por
combo (granularidade, escopo).
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
    CLASSE_POSITIVA,
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
from src.drift.janelas import (
    Janela,
    gerar_janelas_bisemanais,
    gerar_janelas_mensais,
    gerar_repeticoes_aleatorizadas,
)
from src.drift.testes_estatisticos import aplicar_todos


PADRAO_EMBEDDINGS: Path = (
    DIR_ARTEFATOS_DRIFT / "embeddings" / "bertimbau_base_cls" / "embeddings.parquet"
)
PADRAO_CORPUS: Path = DIR_DADOS_PROCESSADO / "corpus_opcao7.parquet"


def carregar_e_alinhar(
    caminho_embeddings: Path, caminho_corpus: Path
) -> pd.DataFrame:
    """
    Lê embeddings + corpus e devolve um único DataFrame ordenado por
    `date`, com colunas `link`, `date`, `y_original`, `embedding`.
    """
    if not caminho_embeddings.exists():
        raise FileNotFoundError(
            f"Embeddings não encontrados em {caminho_embeddings}. "
            "Rode scripts/drift/compute_embeddings.py antes."
        )
    if not caminho_corpus.exists():
        raise FileNotFoundError(
            f"Corpus não encontrado em {caminho_corpus}. "
            "Rode scripts/preprocessar.py antes."
        )

    emb = pd.read_parquet(caminho_embeddings, engine="pyarrow")
    corp = pd.read_parquet(
        caminho_corpus,
        engine="pyarrow",
        columns=["link", "y_original"],
    )
    df = emb.merge(corp, on="link", how="inner", validate="one_to_one")
    if len(df) != len(emb):
        raise ValueError(
            f"Join inconsistente: {len(emb)} embeddings vs {len(df)} após merge. "
            "Verifique se embeddings e corpus vêm da mesma rodada de dedup."
        )
    df = df.sort_values("date", kind="mergesort").reset_index(drop=True)
    return df


def filtrar_por_escopo(df: pd.DataFrame, escopo: str) -> pd.DataFrame:
    """`global` mantém tudo; `mercado` e `nao_mercado` filtram por `y_original`."""
    if escopo == "global":
        return df
    if escopo == "mercado":
        return df[df["y_original"] == CLASSE_POSITIVA].reset_index(drop=True)
    if escopo == "nao_mercado":
        return df[df["y_original"] != CLASSE_POSITIVA].reset_index(drop=True)
    raise ValueError(f"Escopo desconhecido: {escopo!r}. Esperado: {ESCOPOS_DRIFT}.")


def gerar_janelas(df: pd.DataFrame, granularidade: str) -> list[Janela]:
    if granularidade == "mensal":
        return gerar_janelas_mensais(df)
    if granularidade == "bisemanal":
        return gerar_janelas_bisemanais(df)
    raise ValueError(
        f"Granularidade desconhecida: {granularidade!r}. "
        f"Esperado: {GRANULARIDADES_DRIFT}."
    )


def empilhar_embeddings(df: pd.DataFrame, indices: np.ndarray) -> np.ndarray:
    """Concatena os embeddings dos índices num array (n, hidden_dim) float32."""
    return np.stack(df.iloc[indices]["embedding"].values).astype(np.float32)


def comparar_pares_consecutivos(
    embeddings_por_janela: list[np.ndarray],
    rotulos: list[str],
    *,
    condicao: str,
    repeticao_id: int,
    n_permutations: int,
    seed: int,
    device: str | None,
    log_prefixo: str = "",
    limite_pares: int | None = None,
) -> list[dict]:
    """
    Aplica os 4 testes em cada par (i, i+1) consecutivo. Devolve lista
    de dicionários prontos para virar DataFrame.
    """
    n_pares = len(embeddings_por_janela) - 1
    if limite_pares is not None:
        n_pares = min(n_pares, limite_pares)

    linhas: list[dict] = []
    t0 = time.perf_counter()
    for i in range(n_pares):
        x_a = embeddings_por_janela[i]
        x_b = embeddings_por_janela[i + 1]
        # Seed por par mantém determinismo entre execuções repetidas.
        seed_par = seed + i
        resultados = aplicar_todos(
            x_a,
            x_b,
            n_permutations=n_permutations,
            seed=seed_par,
            device=device,
        )
        for nome_teste, r in resultados.items():
            linhas.append(
                {
                    "janela_a": rotulos[i],
                    "janela_b": rotulos[i + 1],
                    "teste": nome_teste,
                    "repeticao": repeticao_id,
                    "condicao": condicao,
                    "p_value": r.p_value,
                    "estatistica": r.estatistica,
                }
            )
        decorrido = time.perf_counter() - t0
        print(
            f"  {log_prefixo}par {i + 1:>3}/{n_pares} "
            f"({rotulos[i]} → {rotulos[i + 1]}) "
            f"[{decorrido / 60:.1f} min]",
            flush=True,
        )
    return linhas


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
        "--n-permutations",
        type=int,
        default=100,
        help="Permutações para KTS/LSDD (default 100; reduzir para smoke).",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda", "gpu"],
        default=None,
        help="Dispositivo para KTS/LSDD (default: auto-detect).",
    )
    parser.add_argument(
        "--limite-pares",
        type=int,
        default=None,
        help="Limita a N pares consecutivos para smoke test.",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    print(f"[1/5] Carregando embeddings e corpus...")
    df = carregar_e_alinhar(args.embeddings, args.corpus)
    print(f"       {len(df):,} artigos alinhados.")

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
        "b1_statistical",
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
    linhas = comparar_pares_consecutivos(
        emb_temporais,
        rotulos_temporais,
        condicao="time_ordered",
        repeticao_id=0,
        n_permutations=args.n_permutations,
        seed=SEED,
        device=args.device,
        log_prefixo="[time-ordered] ",
        limite_pares=args.limite_pares,
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
        # Seed da repetição: SEED + rep_id + N_pares ofereceria mais
        # isolamento; mas para reprodutibilidade simples, usa SEED + rep_id.
        linhas_rep = comparar_pares_consecutivos(
            emb_rand,
            rotulos_rand,
            condicao="randomized",
            repeticao_id=rep_id,
            n_permutations=args.n_permutations,
            seed=SEED + rep_id + 1,
            device=args.device,
            log_prefixo=f"[rand rep={rep_id}] ",
            limite_pares=args.limite_pares,
        )
        linhas.extend(linhas_rep)

    duracao = time.perf_counter() - t_inicio
    print(f"\nGravando resultados em {destino}...")
    df_resultados = pd.DataFrame(linhas)
    salvar_resultados(destino, df_resultados)

    metadata = construir_metadata_drift(
        bloco="b1_statistical",
        escopo=args.escopo,
        granularidade=args.granularidade,
        corpus=args.corpus,
        n_artigos=len(df_filtrado),
        embedding="bertimbau_base_cls",
        embedding_path=args.embeddings,
        janelas=[j.resumir() for j in janelas],
        duracao_segundos=duracao,
        extras={
            "n_permutations": args.n_permutations,
            "device_solicitado": args.device,
            "n_pares_time_ordered": len(janelas) - 1,
            "n_pares_randomized_total": args.n_repeticoes * (len(janelas) - 1),
            "limite_pares": args.limite_pares,
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
