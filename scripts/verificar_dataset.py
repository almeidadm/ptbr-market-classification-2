"""
Verificação de integridade do dataset bruto FolhaUOL.

Dataset: marlesson/news-of-the-site-folhauol
Fonte:   https://www.kaggle.com/datasets/marlesson/news-of-the-site-folhauol

Este script NÃO baixa o dataset — o download é manual, via navegador,
sem exigir credenciais da API do Kaggle. O script apenas valida a
integridade do arquivo baixado e registra seu hash.

Fluxo:
    1. Baixe o arquivo .zip pela URL acima e salve em data/raw/.
       (Nome esperado por padrão: archive.zip — ajustável via --arquivo.)
    2. Rode: python scripts/verificar_dataset.py
    3. No primeiro uso, o SHA256 é calculado e gravado em
       data/raw/DATASET_HASH.txt (TOFU — trust on first use).
       Commit esse arquivo para fixar a referência.
    4. Execuções seguintes comparam o hash atual contra o registrado
       e falham se houver divergência.

Opcional:
    --extrair  extrai o conteúdo do zip em data/raw/ após validar.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import zipfile
from pathlib import Path

RAIZ_PROJETO = Path(__file__).resolve().parent.parent
DIR_DADOS_BRUTOS = RAIZ_PROJETO / "data" / "raw"
ARQUIVO_HASH = DIR_DADOS_BRUTOS / "DATASET_HASH.txt"
NOME_ARQUIVO_PADRAO = "archive.zip"
URL_KAGGLE = "https://www.kaggle.com/datasets/marlesson/news-of-the-site-folhauol"


def calcular_sha256(caminho: Path, tamanho_bloco: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with caminho.open("rb") as f:
        for bloco in iter(lambda: f.read(tamanho_bloco), b""):
            h.update(bloco)
    return h.hexdigest()


def ler_hash_registrado() -> str | None:
    if not ARQUIVO_HASH.exists():
        return None
    for linha in ARQUIVO_HASH.read_text(encoding="utf-8").splitlines():
        linha = linha.strip()
        if linha and not linha.startswith("#"):
            return linha.split()[0]
    return None


def registrar_hash(arquivo_zip: Path, hash_sha256: str) -> None:
    conteudo = (
        "# SHA256 de referência do dataset bruto FolhaUOL.\n"
        "# Gravado no primeiro uso (TOFU) por scripts/verificar_dataset.py.\n"
        "# Commit este arquivo para fixar a referência compartilhada.\n"
        f"{hash_sha256}  {arquivo_zip.name}\n"
    )
    ARQUIVO_HASH.write_text(conteudo, encoding="utf-8")


def extrair(arquivo_zip: Path, destino: Path) -> None:
    with zipfile.ZipFile(arquivo_zip) as zf:
        zf.extractall(destino)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Valida integridade do dataset bruto FolhaUOL em data/raw/.",
    )
    parser.add_argument(
        "--arquivo",
        default=NOME_ARQUIVO_PADRAO,
        help=f"Nome do arquivo em data/raw/ (padrão: {NOME_ARQUIVO_PADRAO})",
    )
    parser.add_argument(
        "--extrair",
        action="store_true",
        help="Após validar o hash, extrair o conteúdo do zip em data/raw/",
    )
    args = parser.parse_args()

    arquivo_zip = DIR_DADOS_BRUTOS / args.arquivo
    if not arquivo_zip.exists():
        print(f"ERRO: arquivo não encontrado: {arquivo_zip}", file=sys.stderr)
        print(
            f"\nBaixe o dataset manualmente em:\n  {URL_KAGGLE}\n"
            f"e salve o arquivo em: {DIR_DADOS_BRUTOS}/",
            file=sys.stderr,
        )
        return 1

    print(f"Calculando SHA256 de {arquivo_zip.name}...")
    hash_calculado = calcular_sha256(arquivo_zip)
    hash_registrado = ler_hash_registrado()

    if hash_registrado is None:
        registrar_hash(arquivo_zip, hash_calculado)
        print(f"SHA256: {hash_calculado}")
        print(
            f"Primeiro uso — hash registrado em {ARQUIVO_HASH.relative_to(RAIZ_PROJETO)}.\n"
            "Commit esse arquivo para fixar a referência."
        )
    elif hash_calculado != hash_registrado:
        print("ERRO: hash do arquivo não bate com o registrado.", file=sys.stderr)
        print(f"  registrado: {hash_registrado}", file=sys.stderr)
        print(f"  calculado:  {hash_calculado}", file=sys.stderr)
        return 2
    else:
        print(f"Hash OK: {hash_calculado}")

    if args.extrair:
        print(f"Extraindo {arquivo_zip.name} em {DIR_DADOS_BRUTOS}/ ...")
        extrair(arquivo_zip, DIR_DADOS_BRUTOS)
        print("Extração concluída.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
