"""
Escrita e leitura dos artefatos de drift (ADR 011 §D.6).

Estrutura canônica:
    artifacts/drift/<bloco>/<YYYYMMDD-HHMM>-<granularidade>-<escopo>/
    ├── metadata.json
    └── results.parquet

Onde `<bloco>` ∈ {`b1_statistical`, `b2_semantic`, `b3_cpd`,
`embeddings`} e `<granularidade>` ∈ {`mensal`, `bisemanal`, `diario`}.
Para o bloco `embeddings`, o diretório é único por modelo (sem
timestamp); para B3, a granularidade fica fixa em `diario` (centróides
diários) conforme ADR 011 §D.2.

Re-execuções: sufixo `-v2`, `-v3`, ... no slug de diretório para não
sobrescrever artefatos anteriores, mesmo padrão de
`src.experimento.artefatos`.
"""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.config import (
    ALPHA_DRIFT,
    DIR_ARTEFATOS_DRIFT,
    N_REPETICOES_DRIFT,
    NOMES_TESTES_B1,
    SEED,
)


BLOCOS_VALIDOS: tuple[str, ...] = (
    "b1_statistical",
    "b2_semantic",
    "b3_cpd",
    "embeddings",
)


def _timestamp_slug(agora: datetime | None = None) -> str:
    agora = agora or datetime.now(timezone.utc)
    return agora.strftime("%Y%m%d-%H%M")


def _slug_execucao(
    granularidade: str | None,
    escopo: str | None,
) -> str:
    partes = [p for p in (granularidade, escopo) if p]
    return "-".join(partes)


def montar_dir_drift(
    bloco: str,
    *,
    escopo: str | None = None,
    granularidade: str | None = None,
    raiz: Path = DIR_ARTEFATOS_DRIFT,
    timestamp: str | None = None,
) -> Path:
    """
    Cria o diretório de execução para um bloco de drift.

    Para `bloco='embeddings'`, o caminho retornado é
    `<raiz>/embeddings/<escopo>` (sem timestamp, porque o cache é
    determinístico por configuração). Para os demais blocos, segue o
    padrão `<raiz>/<bloco>/<timestamp>-<granularidade>-<escopo>` com
    versionamento `-v2`, `-v3`...
    """
    if bloco not in BLOCOS_VALIDOS:
        raise ValueError(
            f"Bloco inválido: {bloco!r}. Esperado: {BLOCOS_VALIDOS}."
        )

    if bloco == "embeddings":
        if escopo is None:
            raise ValueError("Bloco `embeddings` exige `escopo` (nome do modelo).")
        destino = raiz / bloco / escopo
        destino.mkdir(parents=True, exist_ok=True)
        return destino

    if escopo is None:
        raise ValueError(f"Bloco {bloco!r} exige `escopo`.")
    if bloco != "b3_cpd" and granularidade is None:
        raise ValueError(
            f"Bloco {bloco!r} exige `granularidade` "
            "(`b3_cpd` opera apenas em granularidade diária)."
        )

    granularidade_efetiva = granularidade or "diario"
    timestamp = timestamp or _timestamp_slug()
    slug = _slug_execucao(granularidade_efetiva, escopo)
    candidato = raiz / bloco / f"{timestamp}-{slug}"
    versao = 2
    while candidato.exists():
        candidato = raiz / bloco / f"{timestamp}-{slug}-v{versao}"
        versao += 1
    candidato.mkdir(parents=True)
    return candidato


def obter_git_commit() -> str | None:
    """Retorna o SHA do HEAD ou `None` se git não estiver disponível."""
    try:
        saida = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        return None
    if saida.returncode != 0:
        return None
    return saida.stdout.strip() or None


def hash_arquivo(caminho: Path) -> str | None:
    """SHA-256 de um arquivo, ou `None` se inexistente."""
    if not caminho.exists():
        return None
    hasher = hashlib.sha256()
    with caminho.open("rb") as arq:
        for bloco in iter(lambda: arq.read(1 << 20), b""):
            hasher.update(bloco)
    return hasher.hexdigest()


def _versoes_bibliotecas() -> dict[str, str | None]:
    def _versao(nome_modulo: str) -> str | None:
        try:
            modulo = __import__(nome_modulo)
        except ImportError:
            return None
        return getattr(modulo, "__version__", None)

    return {
        "python": platform.python_version(),
        "numpy": _versao("numpy"),
        "pandas": _versao("pandas"),
        "scipy": _versao("scipy"),
        "scikit-learn": _versao("sklearn"),
        "torch": _versao("torch"),
        "transformers": _versao("transformers"),
        "alibi_detect": _versao("alibi_detect"),
        "ruptures": _versao("ruptures"),
    }


def construir_metadata_drift(
    *,
    bloco: str,
    escopo: str,
    granularidade: str,
    corpus: Path,
    n_artigos: int,
    embedding: str,
    embedding_path: Path | None = None,
    janelas: list[dict] | None = None,
    n_repeticoes: int = N_REPETICOES_DRIFT,
    alpha: float = ALPHA_DRIFT,
    tests: tuple[str, ...] | None = None,
    extras: dict | None = None,
    duracao_segundos: float | None = None,
) -> dict:
    """
    Monta o dicionário de metadata serializado em `metadata.json`.

    Campos seguem o schema da ADR 011 §D.6 e são compatíveis com o
    padrão de `src.experimento.artefatos.construir_metadata` em termos
    de git/seed/versões.
    """
    if bloco not in BLOCOS_VALIDOS:
        raise ValueError(
            f"Bloco inválido: {bloco!r}. Esperado: {BLOCOS_VALIDOS}."
        )
    return {
        "adr_referencia": "011",
        "bloco": bloco,
        "granularidade": granularidade,
        "escopo": escopo,
        "timestamp_iso": datetime.now(timezone.utc).isoformat(),
        "git_commit": obter_git_commit(),
        "seed_global": SEED,
        "alpha": alpha,
        "n_repeticoes": n_repeticoes,
        "tests": list(tests) if tests is not None else list(NOMES_TESTES_B1),
        "embedding": embedding,
        "embedding_path": str(embedding_path) if embedding_path else None,
        "embedding_sha256": (
            hash_arquivo(embedding_path) if embedding_path else None
        ),
        "corpus": {
            "caminho": str(corpus),
            "sha256": hash_arquivo(corpus),
            "n_artigos": n_artigos,
        },
        "janelas": janelas,
        "framework_versions": _versoes_bibliotecas(),
        "comando_cli": " ".join(sys.argv) if sys.argv else None,
        "duracao_segundos": duracao_segundos,
        "extras": extras or {},
    }


def salvar_json(caminho: Path, dados: dict) -> None:
    caminho.parent.mkdir(parents=True, exist_ok=True)
    caminho.write_text(
        json.dumps(dados, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )


def carregar_json(caminho: Path) -> dict:
    return json.loads(caminho.read_text(encoding="utf-8"))


def salvar_resultados(dir_execucao: Path, df_resultados: pd.DataFrame) -> Path:
    """
    Grava `results.parquet` na pasta da execução.

    O schema do DataFrame varia por bloco (vide ADR 011 §D.6); esta
    função não valida colunas, deixando essa responsabilidade aos
    callers (módulos B1/B2/B3 e seus testes).
    """
    caminho = dir_execucao / "results.parquet"
    df_resultados.to_parquet(
        caminho, engine="pyarrow", compression="snappy", index=False
    )
    return caminho


def carregar_resultados(dir_execucao: Path) -> pd.DataFrame:
    return pd.read_parquet(dir_execucao / "results.parquet", engine="pyarrow")
