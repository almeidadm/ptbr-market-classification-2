"""
Gera o cache de embeddings BERTimbau [CLS] sobre o corpus pós-dedup
para consumo dos blocos B1, B2 e B3 da detecção de drift (ADRs 010 e
011).

Entrada esperada: `data/processado/corpus_opcao7.parquet` produzido por
`scripts/preprocessar.py`. Colunas mínimas: `link`, `date`, `title`,
`text`.

Saída em `artifacts/drift/embeddings/bertimbau_base_cls/`:
- `metadata.json`: configuração da execução (modelo, pooling, seed,
  hashes, versões).
- `embeddings.parquet`: colunas `link`, `date`, `embedding`
  (list[float32] de 768 dimensões).

Determinismo: `torch.manual_seed(SEED)` + `eval()` (sem dropout).
Reprodutibilidade exata exige mesmo hardware (kernels CUDA fp32 podem
diferir entre arquiteturas).

Variáveis de ambiente honradas:
    PTBR_MC_DIR_DADOS_PROC   default: data/processado
    PTBR_MC_DIR_DRIFT        default: artifacts/drift
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
    BERTIMBAU_FIXOS,
    DIR_ARTEFATOS_DRIFT,
    DIR_DADOS_PROCESSADO,
    MAX_TOKENS_BERTIMBAU,
    SEED,
)
from src.drift.artefatos import (
    construir_metadata_drift,
    montar_dir_drift,
    salvar_json,
)
from src.modelos.texto import juntar_titulo_texto


COLUNAS_OBRIGATORIAS: tuple[str, ...] = ("link", "date", "title", "text")
NOME_ESCOPO_EMBEDDING: str = "bertimbau_base_cls"


def _checar_gpu() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    print(
        "AVISO: nenhuma GPU CUDA detectada. Execução em CPU será ~50× mais "
        "lenta; abortando antes de gastar horas. Use --permitir-cpu para "
        "forçar execução em CPU (não recomendado para o corpus completo).",
        file=sys.stderr,
    )
    return "cpu"


def _carregar_corpus(caminho: Path) -> pd.DataFrame:
    if not caminho.exists():
        raise FileNotFoundError(
            f"Corpus parquet não encontrado em {caminho}. "
            "Rode `scripts/preprocessar.py` antes."
        )
    df = pd.read_parquet(caminho, engine="pyarrow")
    faltando = [c for c in COLUNAS_OBRIGATORIAS if c not in df.columns]
    if faltando:
        raise ValueError(
            f"Corpus em {caminho} sem colunas obrigatórias: {faltando}. "
            f"Disponíveis: {list(df.columns)}."
        )
    return df


def _preparar_textos(df: pd.DataFrame) -> list[str]:
    return [
        juntar_titulo_texto(linha["title"], linha["text"])
        for _, linha in df.iterrows()
    ]


def computar_embeddings(
    textos: list[str],
    *,
    nome_modelo: str,
    max_seq_length: int,
    batch_size: int,
    device: str,
    seed: int,
    log_a_cada: int = 50,
) -> np.ndarray:
    """
    Tokeniza e roda forward pass em lote, extraindo o embedding do
    token `[CLS]` (primeira posição da última camada).

    Retorna array `(n_artigos, hidden_dim)` em float32.
    """
    import torch
    from transformers import AutoModel, AutoTokenizer

    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)

    tokenizer = AutoTokenizer.from_pretrained(nome_modelo)
    modelo = AutoModel.from_pretrained(nome_modelo).to(device)
    modelo.eval()

    saidas: list[np.ndarray] = []
    n_lotes = (len(textos) + batch_size - 1) // batch_size
    inicio_global = time.perf_counter()

    with torch.no_grad():
        for idx_lote, inicio in enumerate(range(0, len(textos), batch_size)):
            lote = textos[inicio : inicio + batch_size]
            tokens = tokenizer(
                lote,
                truncation=True,
                padding=True,
                max_length=max_seq_length,
                return_tensors="pt",
            ).to(device)
            hidden = modelo(**tokens).last_hidden_state
            cls = hidden[:, 0, :].to(torch.float32).cpu().numpy()
            saidas.append(cls)

            if (idx_lote + 1) % log_a_cada == 0 or (idx_lote + 1) == n_lotes:
                decorrido = time.perf_counter() - inicio_global
                ritmo = (idx_lote + 1) * batch_size / max(decorrido, 1e-6)
                eta = max(0.0, (len(textos) - (idx_lote + 1) * batch_size) / ritmo)
                print(
                    f"  lote {idx_lote + 1:>5}/{n_lotes} "
                    f"({ritmo:.0f} artigos/s, ETA {eta / 60:.1f} min)",
                    flush=True,
                )

    return np.vstack(saidas)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus",
        type=Path,
        default=Path(
            os.environ.get("PTBR_MC_CORPUS_DRIFT")
            or DIR_DADOS_PROCESSADO / "corpus_opcao7.parquet"
        ),
        help="Parquet pós-dedup (default: data/processado/corpus_opcao7.parquet).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DIR_ARTEFATOS_DRIFT,
        help="Raiz de artifacts/drift (default: artifacts/drift).",
    )
    parser.add_argument(
        "--modelo",
        type=str,
        default=BERTIMBAU_FIXOS["model_name"],
        help="Checkpoint HuggingFace (default: neuralmind/bert-base-portuguese-cased).",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=MAX_TOKENS_BERTIMBAU,
        help=f"Comprimento máximo em tokens (default: {MAX_TOKENS_BERTIMBAU}).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch de inferência (default: 64).",
    )
    parser.add_argument(
        "--permitir-cpu",
        action="store_true",
        help="Roda em CPU se GPU não estiver disponível (lento; sem essa "
        "flag o script aborta se cuda não estiver presente).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Sobrescreve embeddings.parquet existente.",
    )
    parser.add_argument(
        "--limite-artigos",
        type=int,
        default=None,
        help="Limita a N artigos para smoke test (default: todos).",
    )
    args = parser.parse_args(argv)

    device = _checar_gpu()
    if device == "cpu" and not args.permitir_cpu:
        return 2

    print(f"[1/4] Carregando corpus de {args.corpus}...")
    df = _carregar_corpus(args.corpus)
    if args.limite_artigos is not None:
        df = df.head(args.limite_artigos).copy()
    print(f"       {len(df):,} artigos.")

    destino = montar_dir_drift(
        "embeddings",
        escopo=NOME_ESCOPO_EMBEDDING,
        raiz=args.out,
    )
    caminho_parquet = destino / "embeddings.parquet"
    if caminho_parquet.exists() and not args.force:
        print(
            f"ERRO: {caminho_parquet} já existe. Use --force para sobrescrever.",
            file=sys.stderr,
        )
        return 3

    print(f"[2/4] Preparando textos ({len(df):,} entradas)...")
    textos = _preparar_textos(df)

    print(
        f"[3/4] Computando embeddings com {args.modelo} "
        f"(batch={args.batch_size}, max_len={args.max_seq_length}, device={device})..."
    )
    t0 = time.perf_counter()
    embeddings = computar_embeddings(
        textos,
        nome_modelo=args.modelo,
        max_seq_length=args.max_seq_length,
        batch_size=args.batch_size,
        device=device,
        seed=SEED,
    )
    duracao = time.perf_counter() - t0
    print(
        f"       embeddings shape={embeddings.shape}, "
        f"tempo={duracao / 60:.1f} min."
    )

    print(f"[4/4] Persistindo em {destino}...")
    df_out = pd.DataFrame(
        {
            "link": df["link"].to_numpy(),
            "date": df["date"].to_numpy(),
            "embedding": list(embeddings.astype(np.float32)),
        }
    )
    df_out.to_parquet(
        caminho_parquet, engine="pyarrow", compression="snappy", index=False
    )

    metadata = construir_metadata_drift(
        bloco="embeddings",
        escopo=NOME_ESCOPO_EMBEDDING,
        granularidade="n_a",
        corpus=args.corpus,
        n_artigos=len(df),
        embedding=NOME_ESCOPO_EMBEDDING,
        embedding_path=caminho_parquet,
        duracao_segundos=duracao,
        extras={
            "modelo": args.modelo,
            "pooling": "cls",
            "max_seq_length": args.max_seq_length,
            "batch_size": args.batch_size,
            "device": device,
            "hidden_dim": int(embeddings.shape[1]),
        },
    )
    salvar_json(destino / "metadata.json", metadata)

    tamanho_mb = caminho_parquet.stat().st_size / 1024**2
    print(
        f"       {caminho_parquet.name} ({tamanho_mb:.1f} MB) + metadata.json"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
