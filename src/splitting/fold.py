"""
Tipo `Fold` compartilhado pelos geradores de partição.

Cada fold carrega quatro subconjuntos de índices posicionais no
DataFrame de entrada:

- `treino_idx` / `teste_idx`: partição externa (avaliação).
- `inner_train_idx` / `inner_val_idx`: partição interna sobre o treino,
  consumida pela busca de hiperparâmetros (ADR 009 §D.2).

`inicio_teste` e `fim_teste` são preenchidos pelo protocolo rolling-origin
(ADR 008 primário) e ficam como `None` no k-fold estratificado.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Fold:
    indice: int
    treino_idx: np.ndarray
    teste_idx: np.ndarray
    inner_train_idx: np.ndarray
    inner_val_idx: np.ndarray
    inicio_teste: pd.Timestamp | None
    fim_teste: pd.Timestamp | None

    def resumir(self) -> dict:
        """Resumo serializável para `metadata.json`."""
        return {
            "fold": self.indice,
            "n_treino": int(len(self.treino_idx)),
            "n_teste": int(len(self.teste_idx)),
            "n_inner_train": int(len(self.inner_train_idx)),
            "n_inner_val": int(len(self.inner_val_idx)),
            "inicio_teste": (
                self.inicio_teste.isoformat() if self.inicio_teste is not None else None
            ),
            "fim_teste": (
                self.fim_teste.isoformat() if self.fim_teste is not None else None
            ),
        }
