"""
Testes para `src.modelos.bertimbau`.

Os testes de treino e predição exigem GPU — são skipados quando CUDA
não está disponível. O teste de grid é trivial e sempre roda.
"""
from __future__ import annotations

import importlib.util

import pandas as pd
import pytest

from src.config import GRID_BERTIMBAU_EPOCHS, GRID_BERTIMBAU_LR
from src.modelos.bertimbau import grid_bertimbau


def _ambiente_pronto() -> bool:
    """True apenas se GPU + todas as libs pesadas estiverem disponíveis."""
    for nome in ("torch", "transformers", "datasets"):
        if importlib.util.find_spec(nome) is None:
            return False
    import torch
    return torch.cuda.is_available()


def test_grid_tem_seis_configuracoes() -> None:
    grid = grid_bertimbau()
    assert len(grid) == len(GRID_BERTIMBAU_LR) * len(GRID_BERTIMBAU_EPOCHS)
    for cfg in grid:
        assert "learning_rate" in cfg
        assert "num_train_epochs" in cfg


@pytest.mark.skipif(
    not _ambiente_pronto(),
    reason="Treino do BERTimbau exige GPU CUDA + transformers + datasets.",
)
def test_smoke_treino_e_predicao_em_subset_pequeno(tmp_path) -> None:
    from src.modelos.bertimbau import predizer_bertimbau, treinar_bertimbau

    df = pd.DataFrame(
        {
            "title": ["notícia " + str(i) for i in range(32)],
            "text": ["corpo " + str(i) for i in range(32)],
            "y_colapsado": (["mercado", "outros"] * 16),
        }
    )
    modelo, tokenizer, label2id = treinar_bertimbau(
        df.iloc[:20],
        df.iloc[20:26],
        hp={"learning_rate": 3e-5, "num_train_epochs": 1},
        classes=["mercado", "outros"],
        dir_saida=tmp_path / "bert",
    )
    y_pred, probs, classes = predizer_bertimbau(modelo, tokenizer, df.iloc[26:], label2id)
    assert len(y_pred) == 6
    assert probs.shape == (6, 2)
