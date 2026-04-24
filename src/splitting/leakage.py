"""
Diagnóstico de near-duplicate leakage entre treino e teste (ADR 008).

Para cada fold, conta pares `(id_a, id_b)` registrados no artefato
`artifacts/eda/tabelas/07-near-duplicates-pares.csv` que:

- Caem inteiramente dentro do treino (neutro).
- Caem inteiramente dentro do teste (neutro).
- Cruzam treino↔teste (**vazamento**).

A fração `pares_cruzando / pares_presentes` é a métrica reportada por
fold e globalmente. Pares cujos dois lados estão fora do fold (nenhum
em treino ou teste) são descartados.

ADR 007 impõe a preservação (não mitigação) desses pares — o número
aqui funciona apenas como diagnóstico.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.splitting.fold import Fold


COLUNAS_CSV = ("id_a", "id_b")


def carregar_pares_near_dup(caminho: Path) -> pd.DataFrame:
    """Lê o CSV da EDA, mantendo apenas as colunas essenciais."""
    df = pd.read_csv(caminho)
    faltando = [c for c in COLUNAS_CSV if c not in df.columns]
    if faltando:
        raise ValueError(
            f"CSV de near-dups sem colunas esperadas: {faltando} (em {caminho})."
        )
    return df[list(COLUNAS_CSV)].copy()


def diagnosticar_leakage(
    folds: list[Fold],
    df_corpus: pd.DataFrame,
    pares_near_dup: pd.DataFrame,
) -> dict:
    """
    Agrega o diagnóstico por fold e o resumo global.

    `df_corpus` deve conter `id_raw` — a posição no corpus bruto que
    serve de chave comum com `pares_near_dup`. Pares cujos dois `id_*`
    não estão em `df_corpus["id_raw"]` (ex.: uma das linhas foi removida
    pela dedup ou pelo recorte) são descartados globalmente.
    """
    if "id_raw" not in df_corpus.columns:
        raise ValueError(
            "`df_corpus` precisa da coluna `id_raw` produzida por "
            "`src.preprocessamento.carregamento.preparar_corpus`."
        )

    id_raw_para_posicao = {
        int(id_raw): posicao
        for posicao, id_raw in enumerate(df_corpus["id_raw"].to_numpy())
    }

    pares_validos: list[tuple[int, int]] = []
    for id_a, id_b in pares_near_dup[["id_a", "id_b"]].to_numpy():
        a = id_raw_para_posicao.get(int(id_a))
        b = id_raw_para_posicao.get(int(id_b))
        if a is None or b is None:
            continue
        pares_validos.append((a, b))

    por_fold: list[dict] = []
    total_cruzando = 0
    total_presentes = 0

    for fold in folds:
        treino = set(map(int, fold.treino_idx.tolist()))
        teste = set(map(int, fold.teste_idx.tolist()))

        pares_presentes = 0
        pares_cruzando = 0
        pares_dentro_treino = 0
        pares_dentro_teste = 0

        for a, b in pares_validos:
            em_treino_a, em_teste_a = a in treino, a in teste
            em_treino_b, em_teste_b = b in treino, b in teste
            lado_a = "treino" if em_treino_a else ("teste" if em_teste_a else None)
            lado_b = "treino" if em_treino_b else ("teste" if em_teste_b else None)
            if lado_a is None or lado_b is None:
                continue
            pares_presentes += 1
            if lado_a != lado_b:
                pares_cruzando += 1
            elif lado_a == "treino":
                pares_dentro_treino += 1
            else:
                pares_dentro_teste += 1

        fracao = pares_cruzando / pares_presentes if pares_presentes else 0.0
        por_fold.append(
            {
                "fold": fold.indice,
                "pares_presentes": pares_presentes,
                "pares_cruzando_treino_teste": pares_cruzando,
                "pares_dentro_treino": pares_dentro_treino,
                "pares_dentro_teste": pares_dentro_teste,
                "fracao_cruzando": fracao,
            }
        )
        total_cruzando += pares_cruzando
        total_presentes += pares_presentes

    resumo = {
        "n_pares_validos": len(pares_validos),
        "n_pares_csv_bruto": int(len(pares_near_dup)),
        "agregado": {
            "pares_presentes": total_presentes,
            "pares_cruzando_treino_teste": total_cruzando,
            "fracao_cruzando": (
                total_cruzando / total_presentes if total_presentes else 0.0
            ),
        },
        "por_fold": por_fold,
    }
    return resumo
