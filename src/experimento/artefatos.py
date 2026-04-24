"""
Escrita e leitura dos artefatos de experimento (ADR 009 §D.6).

Estrutura canônica:
    artifacts/experimentos/<YYYYMMDD-HHMM>-<familia>-<recorte>-<protocolo>-<regime>/
    ├── metadata.json
    ├── folds/fold_<i>/{best_hp.json, hp_search.csv, predictions.parquet, metrics.json}
    ├── summary.json
    ├── leakage_diagnostic.json
    └── run.log

Re-execuções: sufixo `-v2`, `-v3`, ... no slug de diretório para não
sobrescrever artefatos anteriores.

Persistência fraca (§D decisão D3): nenhum checkpoint binário — apenas
metadados, predições e métricas. Um re-treino determinístico reproduz.
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

from src.config import DIR_ARTEFATOS_EXPERIMENTOS, SEED


def _timestamp_slug(agora: datetime | None = None) -> str:
    agora = agora or datetime.now(timezone.utc)
    return agora.strftime("%Y%m%d-%H%M")


def _slug_cenario(familia: str, recorte: str, protocolo: str, regime: str) -> str:
    return f"{familia}-{recorte}-{protocolo}-{regime}"


def montar_dir_experimento(
    familia: str,
    recorte: str,
    protocolo: str,
    regime: str,
    raiz: Path = DIR_ARTEFATOS_EXPERIMENTOS,
    timestamp: str | None = None,
) -> Path:
    """
    Cria o diretório do experimento aplicando versionamento via sufixo
    `-v2`, `-v3`... quando um diretório com o mesmo timestamp+slug já
    existe (raro, mas acontece em re-execuções no mesmo minuto).
    """
    timestamp = timestamp or _timestamp_slug()
    slug = _slug_cenario(familia, recorte, protocolo, regime)
    candidato = raiz / f"{timestamp}-{slug}"
    versao = 2
    while candidato.exists():
        candidato = raiz / f"{timestamp}-{slug}-v{versao}"
        versao += 1
    candidato.mkdir(parents=True)
    (candidato / "folds").mkdir()
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


def _hash_arquivo(caminho: Path) -> str | None:
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
        "scikit-learn": _versao("sklearn"),
        "torch": _versao("torch"),
        "transformers": _versao("transformers"),
        "vllm": _versao("vllm"),
    }


def construir_metadata(
    *,
    familia: str,
    recorte: str,
    protocolo: str,
    regime: str,
    n_classes_treino: int,
    n_folds: int,
    hp_search_space: dict,
    hp_por_fold: list[dict],
    corpus_raw: Path,
    corpus_processado: Path,
    n_artigos: int,
    prompt_artifact: Path | None = None,
    modo_uso: str = "train",
    extras: dict | None = None,
) -> dict:
    metadata = {
        "experiment_id": _slug_cenario(familia, recorte, protocolo, regime),
        "timestamp_iso": datetime.now(timezone.utc).isoformat(),
        "git_commit": obter_git_commit(),
        "seed": SEED,
        "familia_modelo": familia,
        "modo_uso": modo_uso,
        "recorte": recorte,
        "protocolo_split": protocolo,
        "regime": regime,
        "n_classes_treino": n_classes_treino,
        "n_folds": n_folds,
        "corpus": {
            "caminho_raw": str(corpus_raw),
            "raw_sha256": _hash_arquivo(corpus_raw),
            "caminho_processado": str(corpus_processado),
            "pos_dedup_sha256": _hash_arquivo(corpus_processado),
            "n_artigos": n_artigos,
        },
        "hp_search_space": hp_search_space,
        "hp_per_fold": hp_por_fold,
        "prompt_artifact": str(prompt_artifact) if prompt_artifact else None,
        "prompt_sha256": _hash_arquivo(prompt_artifact) if prompt_artifact else None,
        "framework_versions": _versoes_bibliotecas(),
        "comando_cli": " ".join(sys.argv) if sys.argv else None,
    }
    if extras:
        metadata["extras"] = extras
    return metadata


def salvar_json(caminho: Path, dados: dict) -> None:
    caminho.parent.mkdir(parents=True, exist_ok=True)
    caminho.write_text(
        json.dumps(dados, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )


def salvar_predictions(
    dir_fold: Path,
    df_teste: pd.DataFrame,
    y_pred,
    scores,
    classes: list[str],
) -> Path:
    """
    Persiste predições por artigo em `predictions.parquet`.
    Colunas: `link, date, y_original, y_colapsado, y_pred, prob_<classe>..., eh_duplicata_text`.
    """
    import numpy as np

    bloco = {
        "link": df_teste["link"].to_numpy(),
        "date": df_teste["date"].to_numpy(),
        "y_original": df_teste["y_original"].to_numpy(),
        "y_colapsado": df_teste["y_colapsado"].to_numpy(),
        "y_pred": np.asarray(y_pred),
        "eh_duplicata_text": df_teste["eh_duplicata_text"].to_numpy(),
    }
    scores = np.asarray(scores)
    for idx, classe in enumerate(classes):
        bloco[f"score_{classe}"] = scores[:, idx]

    caminho = dir_fold / "predictions.parquet"
    pd.DataFrame(bloco).to_parquet(
        caminho, engine="pyarrow", compression="snappy", index=False
    )
    return caminho


def salvar_hp_search(dir_fold: Path, tabela: pd.DataFrame) -> Path:
    caminho = dir_fold / "hp_search.csv"
    tabela.to_csv(caminho, index=False)
    return caminho


def carregar_best_hp_do_primario(caminho_primario: Path, fold_idx: int) -> dict:
    """Lê `folds/fold_<idx>/best_hp.json` do experimento primário."""
    caminho = caminho_primario / "folds" / f"fold_{fold_idx}" / "best_hp.json"
    if not caminho.exists():
        raise FileNotFoundError(
            f"`best_hp.json` não encontrado em {caminho} — o experimento "
            f"primário precisa ter sido executado antes da ablation."
        )
    return json.loads(caminho.read_text(encoding="utf-8"))
