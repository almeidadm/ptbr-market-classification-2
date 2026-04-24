"""
Orquestra o preprocessamento do corpus FolhaUOL (Fase 1 da pipeline).

Pipeline:
1. Carrega `articles.csv` e filtra linhas com `date` ou `text` nulos.
2. Deduplica por `text` exato (ADR 007), preservando `eh_duplicata_text`.
3. Aplica os recortes Opção 7, Opção 4 e Opção 3 (ADR 004).
4. Persiste um parquet por recorte em `data/processado/` e tabela de
   contagens da Opção 4 para revisão do prompt v1 definitivo.
5. Grava hashes dos parquets em `data/processado/hashes.json` para
   consumo pelo runner (ADR 009 §D.6, campo `corpus.pos_dedup_sha256`).

Variáveis de ambiente:
    CAMINHO_CSV          default: data/raw/articles.csv
    DIR_DADOS_PROC       default: data/processado
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

RAIZ_PROJETO = Path(__file__).resolve().parent.parent
if str(RAIZ_PROJETO) not in sys.path:
    sys.path.insert(0, str(RAIZ_PROJETO))

from src.config import (
    ARQUIVO_CSV_RAW,
    DIR_DADOS_PROCESSADO,
    RECORTES,
)
from src.preprocessamento.carregamento import preparar_corpus
from src.preprocessamento.deduplicacao import deduplicar_por_texto_exato
from src.preprocessamento.enumeracao import (
    enumerar_classes_opcao_4,
    tabular_contagens_opcao_4,
)
from src.preprocessamento.recortes import (
    aplicar_opcao_3,
    aplicar_opcao_4,
    aplicar_opcao_7,
)


FUNCOES_RECORTE = {
    "opcao7": aplicar_opcao_7,
    "opcao4": aplicar_opcao_4,
    "opcao3": aplicar_opcao_3,
}


def hash_sha256(caminho: Path) -> str:
    hasher = hashlib.sha256()
    with caminho.open("rb") as arquivo:
        for bloco in iter(lambda: arquivo.read(1 << 20), b""):
            hasher.update(bloco)
    return hasher.hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path(os.environ.get("CAMINHO_CSV", ARQUIVO_CSV_RAW)),
        help="Caminho para articles.csv bruto.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(os.environ.get("DIR_DADOS_PROC", DIR_DADOS_PROCESSADO)),
        help="Diretório de saída para os parquets processados.",
    )
    parser.add_argument(
        "--recortes",
        nargs="+",
        choices=list(RECORTES),
        default=list(RECORTES),
        help="Subconjunto de recortes a materializar.",
    )
    args = parser.parse_args(argv)

    if not args.csv.exists():
        print(f"ERRO: CSV bruto não encontrado em {args.csv}", file=sys.stderr)
        return 1

    args.out.mkdir(parents=True, exist_ok=True)

    print(f"[1/4] Carregando corpus de {args.csv}...")
    df = preparar_corpus(args.csv)
    print(f"       {len(df):,} artigos após filtrar date/text nulos.")

    print("[2/4] Deduplicando por texto exato...")
    df_dedup = deduplicar_por_texto_exato(df)
    n_removidos = len(df) - len(df_dedup)
    print(
        f"       {len(df_dedup):,} artigos após dedup "
        f"({n_removidos:,} removidos; "
        f"{int(df_dedup['eh_duplicata_text'].sum()):,} marcados como eh_duplicata_text=True)."
    )

    hashes: dict[str, str] = {"csv_raw_sha256": hash_sha256(args.csv)}

    if "opcao4" in args.recortes:
        print("[3/4] Enumerando classes da Opção 4...")
        classes_op4 = enumerar_classes_opcao_4(df_dedup)
        tabela_op4 = tabular_contagens_opcao_4(df_dedup)
        caminho_enum = args.out / "enumeracao_opcao4.json"
        caminho_enum.write_text(
            json.dumps(
                {"classes": classes_op4, "n_classes": len(classes_op4)},
                indent=2,
                ensure_ascii=False,
            )
        )
        caminho_tab = args.out / "enumeracao_opcao4_contagens.csv"
        tabela_op4.to_csv(caminho_tab, index=False)
        print(
            f"       {len(classes_op4)} classes enumeradas → {caminho_enum.name}, "
            f"contagens → {caminho_tab.name}"
        )

    print("[4/4] Aplicando recortes e persistindo parquets...")
    for recorte in args.recortes:
        funcao = FUNCOES_RECORTE[recorte]
        df_rec = funcao(df_dedup)
        caminho_parquet = args.out / f"corpus_{recorte}.parquet"
        df_rec.to_parquet(
            caminho_parquet,
            engine="pyarrow",
            compression="snappy",
            index=False,
        )
        classes = sorted(df_rec["y_colapsado"].unique())
        hashes[f"corpus_{recorte}_sha256"] = hash_sha256(caminho_parquet)
        print(
            f"       {recorte}: {len(df_rec):,} artigos, "
            f"{len(classes)} classes → {caminho_parquet.name}"
        )

    caminho_hashes = args.out / "hashes.json"
    caminho_hashes.write_text(json.dumps(hashes, indent=2))
    print(f"Hashes gravados em {caminho_hashes.name}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
