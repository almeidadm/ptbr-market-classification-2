"""
Família BERTimbau base fine-tuned (ADR 009 §D.1, §D.3).

Modelo: `neuralmind/bert-base-portuguese-cased`.
Grid de 6 (`learning_rate × num_train_epochs`). Hiperparâmetros fixos
conforme ADR 009: batch=32, warmup_ratio=0.1, weight_decay=0.01,
max_seq_length=256, optimizer=AdamW, scheduler linear.

Implementação usa `transformers.Trainer` por simplicidade e
replicabilidade. GPU é obrigatória (L4 confirmada no ciclo 2026-04-24).
O módulo é importável sem GPU; o treino levanta erro amigável.
"""
from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
    BERTIMBAU_FIXOS,
    GRID_BERTIMBAU_EPOCHS,
    GRID_BERTIMBAU_LR,
    SEED,
)
from src.modelos.texto import juntar_titulo_texto


def grid_bertimbau() -> list[dict]:
    return [
        {"learning_rate": lr, "num_train_epochs": epochs}
        for lr, epochs in itertools.product(
            GRID_BERTIMBAU_LR, GRID_BERTIMBAU_EPOCHS
        )
    ]


def _preparar_textos(df: pd.DataFrame) -> list[str]:
    return [
        juntar_titulo_texto(linha["title"], linha["text"])
        for _, linha in df.iterrows()
    ]


def _checar_gpu() -> None:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "BERTimbau requer GPU CUDA (L4 confirmada no ciclo 2026-04-24); "
            "nenhuma GPU detectada."
        )


def treinar_bertimbau(
    df_treino: pd.DataFrame,
    df_val: pd.DataFrame,
    hp: dict,
    classes: list[str],
    dir_saida: Path,
) -> tuple[object, object, dict[str, int]]:
    """
    Fine-tuning de uma configuração. Retorna `(model, tokenizer, label2id)`.

    Não persiste o checkpoint fora de `dir_saida` — limpeza fica por
    conta do runner (ADR 009 §D: persistência fraca).
    """
    _checar_gpu()

    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        Trainer,
        TrainingArguments,
    )
    from datasets import Dataset

    label2id = {classe: idx for idx, classe in enumerate(classes)}
    id2label = {idx: classe for classe, idx in label2id.items()}

    tokenizer = AutoTokenizer.from_pretrained(BERTIMBAU_FIXOS["model_name"])

    def _para_dataset(df: pd.DataFrame) -> Dataset:
        return Dataset.from_dict(
            {
                "text": _preparar_textos(df),
                "label": [label2id[y] for y in df["y_colapsado"]],
            }
        )

    ds_treino = _para_dataset(df_treino)
    ds_val = _para_dataset(df_val)

    def tokenizar(exemplo):
        return tokenizer(
            exemplo["text"],
            truncation=True,
            max_length=BERTIMBAU_FIXOS["max_seq_length"],
        )

    ds_treino = ds_treino.map(tokenizar, batched=True, remove_columns=["text"])
    ds_val = ds_val.map(tokenizar, batched=True, remove_columns=["text"])

    modelo = AutoModelForSequenceClassification.from_pretrained(
        BERTIMBAU_FIXOS["model_name"],
        num_labels=len(classes),
        id2label=id2label,
        label2id=label2id,
    )

    argumentos = TrainingArguments(
        output_dir=str(dir_saida),
        per_device_train_batch_size=BERTIMBAU_FIXOS["per_device_train_batch_size"],
        per_device_eval_batch_size=BERTIMBAU_FIXOS["per_device_train_batch_size"] * 2,
        learning_rate=hp["learning_rate"],
        num_train_epochs=hp["num_train_epochs"],
        warmup_ratio=BERTIMBAU_FIXOS["warmup_ratio"],
        weight_decay=BERTIMBAU_FIXOS["weight_decay"],
        optim=BERTIMBAU_FIXOS["optim"],
        lr_scheduler_type=BERTIMBAU_FIXOS["lr_scheduler_type"],
        seed=SEED,
        data_seed=SEED,
        logging_strategy="epoch",
        save_strategy="no",
        eval_strategy="no",
        report_to=[],
        bf16=True,
    )

    trainer = Trainer(
        model=modelo,
        args=argumentos,
        train_dataset=ds_treino,
        eval_dataset=ds_val,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    )
    trainer.train()
    return modelo, tokenizer, label2id


def predizer_bertimbau(
    modelo,
    tokenizer,
    df_teste: pd.DataFrame,
    label2id: dict[str, int],
    batch_size: int = 64,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Predição em lote. Retorna `(y_pred, probs, classes)` com `classes`
    na mesma ordem de `label2id`.
    """
    _checar_gpu()

    import torch

    modelo.eval()
    id2label = {idx: classe for classe, idx in label2id.items()}
    classes = [id2label[i] for i in range(len(id2label))]

    textos = _preparar_textos(df_teste)
    probs_lista: list[np.ndarray] = []
    with torch.no_grad():
        for inicio in range(0, len(textos), batch_size):
            lote = textos[inicio : inicio + batch_size]
            tokens = tokenizer(
                lote,
                truncation=True,
                padding=True,
                max_length=BERTIMBAU_FIXOS["max_seq_length"],
                return_tensors="pt",
            ).to(modelo.device)
            logits = modelo(**tokens).logits
            probs_lote = torch.softmax(logits, dim=-1).cpu().numpy()
            probs_lista.append(probs_lote)

    probs = np.vstack(probs_lista)
    y_pred_idx = probs.argmax(axis=1)
    y_pred = np.array([classes[i] for i in y_pred_idx])
    return y_pred, probs, classes
