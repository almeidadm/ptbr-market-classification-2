"""
Orquestrador dos experimentos (ADR 009).

Responsabilidades:
1. Carregar o corpus processado do recorte.
2. Gerar os folds conforme o protocolo.
3. Para cada fold, aplicar regra (ii) da ADR 009 §D.2:
   - Primário (opcao7 × rolling × multi): busca de HPs por fold.
   - Ablations: reusam `best_hp.json` do fold 0 do primário.
4. Treinar, predizer, calcular métricas e persistir artefatos.
5. Agregar cross-fold em `summary.json`.
6. Diagnosticar leakage near-dup e gravar `leakage_diagnostic.json`.

O runner é uma função — sem classes, sem abstrações. Adaptação por
família vive em `_treinar_e_predizer`.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
    ARQUIVO_CSV_RAW,
    DIR_ARTEFATOS_EXPERIMENTOS,
    DIR_ARTEFATOS_PROMPTS,
    DIR_DADOS_PROCESSADO,
)
from src.experimento.artefatos import (
    carregar_best_hp_do_primario,
    construir_metadata,
    montar_dir_experimento,
    salvar_hp_search,
    salvar_json,
    salvar_predictions,
)
from src.experimento.metricas import (
    calcular_metricas_fold,
    projetar_binario,
    resumir_cross_fold,
)
from src.modelos import logreg as modelo_logreg
from src.modelos import svm as modelo_svm
from src.modelos.texto import preparar_entrada
from src.splitting.fold import Fold
from src.splitting.kfold_stratified import gerar_folds_kfold
from src.splitting.leakage import carregar_pares_near_dup, diagnosticar_leakage
from src.splitting.rolling_origin import gerar_folds_rolling


PRIMARIO = {"recorte": "opcao7", "protocolo": "rolling", "regime": "multi"}

CAMINHO_PARES_NEAR_DUP = (
    Path(__file__).resolve().parent.parent.parent
    / "artifacts"
    / "eda"
    / "tabelas"
    / "07-near-duplicates-pares.csv"
)

PROMPT_POR_CENARIO = {
    ("opcao7", "multi"): "mercado-classification-zero-opcao7-v1.md",
    ("opcao4", "multi"): "mercado-classification-zero-opcao4-v1.md",
    ("opcao3", "multi"): "mercado-classification-zero-opcao3-v1.md",
    ("opcao7", "bin"): "mercado-classification-zero-binario-v1.md",
}


def eh_primario(recorte: str, protocolo: str, regime: str) -> bool:
    return (
        recorte == PRIMARIO["recorte"]
        and protocolo == PRIMARIO["protocolo"]
        and regime == PRIMARIO["regime"]
    )


def _caminho_corpus(recorte: str, dir_proc: Path) -> Path:
    return dir_proc / f"corpus_{recorte}.parquet"


def _projetar_df_binario(df: pd.DataFrame) -> pd.DataFrame:
    """Converte `y_colapsado` em `{mercado, nao-mercado}`."""
    df = df.copy()
    df["y_colapsado"] = np.where(
        df["y_colapsado"].values == "mercado", "mercado", "nao-mercado"
    )
    return df


def _gerar_folds(df: pd.DataFrame, protocolo: str) -> list[Fold]:
    if protocolo == "rolling":
        return gerar_folds_rolling(df)
    if protocolo == "kfold":
        return gerar_folds_kfold(df)
    raise ValueError(f"Protocolo desconhecido: {protocolo}")


def _grid_familia(familia: str) -> list[dict]:
    if familia == "logreg":
        return modelo_logreg.grid_logreg()
    if familia == "svm":
        return modelo_svm.grid_svm()
    if familia == "bertimbau":
        from src.modelos.bertimbau import grid_bertimbau

        return grid_bertimbau()
    if familia == "llama31-8b-zs":
        return [{"prompt": "fixo-por-recorte"}]
    raise ValueError(f"Família desconhecida: {familia}")


def _treinar_e_predizer(
    familia: str,
    df_treino: pd.DataFrame,
    df_teste: pd.DataFrame,
    hp: dict,
    classes_treino: list[str],
    dir_tmp: Path,
    motor_llama=None,
) -> tuple[np.ndarray, np.ndarray, list[str], dict]:
    """Despacha para a família e retorna `(y_pred, scores, classes, extras)`."""
    y_treino = df_treino["y_colapsado"]

    if familia == "logreg":
        x_tr = preparar_entrada(df_treino)
        x_te = preparar_entrada(df_teste)
        modelo = modelo_logreg.treinar_logreg(x_tr, y_treino, hp)
        y_pred, scores, classes = modelo_logreg.predizer_logreg(modelo, x_te)
        return y_pred, scores, classes, {}

    if familia == "svm":
        x_tr = preparar_entrada(df_treino)
        x_te = preparar_entrada(df_teste)
        modelo = modelo_svm.treinar_svm(x_tr, y_treino, hp)
        y_pred, scores, classes = modelo_svm.predizer_svm(modelo, x_te)
        return y_pred, scores, classes, {}

    if familia == "bertimbau":
        from src.modelos.bertimbau import predizer_bertimbau, treinar_bertimbau

        modelo, tokenizer, label2id = treinar_bertimbau(
            df_treino=df_treino,
            df_val=df_treino.iloc[: max(32, int(0.05 * len(df_treino)))],
            hp=hp,
            classes=classes_treino,
            dir_saida=dir_tmp,
        )
        y_pred, scores, classes = predizer_bertimbau(
            modelo, tokenizer, df_teste, label2id
        )
        return y_pred, scores, classes, {}

    if familia == "llama31-8b-zs":
        from src.modelos import llama_zero

        if motor_llama is None:
            raise RuntimeError(
                "Motor vLLM precisa ser criado pelo runner e passado para "
                "`_treinar_e_predizer` — não pode ser inicializado por fold."
            )
        textos = llama_zero.preparar_entrada_llama(df_teste).tolist()
        classes = classes_treino
        fallback = llama_zero.classe_majoritaria_do_fold(y_treino)
        prompt = llama_zero.carregar_prompt(hp["prompt_path"])
        y_pred, off_label = llama_zero.classificar_llama_zero(
            textos=textos,
            prompt=prompt,
            classes=classes,
            fallback=fallback,
            motor_vllm=motor_llama,
        )
        # Scores sintéticos para manter o contrato (um-quente baseado em y_pred).
        scores = np.zeros((len(y_pred), len(classes)))
        idx_por_classe = {c: i for i, c in enumerate(classes)}
        for i, p in enumerate(y_pred):
            scores[i, idx_por_classe[p]] = 1.0
        return np.asarray(y_pred), scores, classes, {"off_label_rate": off_label}

    raise ValueError(f"Família desconhecida: {familia}")


def _buscar_melhor_hp(
    familia: str,
    df_inner_train: pd.DataFrame,
    df_inner_val: pd.DataFrame,
    classes_treino: list[str],
    dir_tmp: Path,
    motor_llama=None,
) -> tuple[dict, pd.DataFrame]:
    """
    Percorre o grid, treina em `inner_train`, avalia em `inner_val` pela
    métrica primária (F1 binária projetada de `mercado`) e seleciona o
    argmax. Retorna `(melhor_hp, tabela_de_busca)`.
    """
    grid = _grid_familia(familia)
    y_val_bin = projetar_binario(df_inner_val["y_colapsado"].to_numpy())

    registros: list[dict] = []
    melhor_hp: dict | None = None
    melhor_f1 = -1.0

    for hp in grid:
        y_pred, _, _, _ = _treinar_e_predizer(
            familia=familia,
            df_treino=df_inner_train,
            df_teste=df_inner_val,
            hp=hp,
            classes_treino=classes_treino,
            dir_tmp=dir_tmp,
            motor_llama=motor_llama,
        )
        y_pred_bin = projetar_binario(y_pred)
        tp = int(np.sum((y_val_bin == 1) & (y_pred_bin == 1)))
        fp = int(np.sum((y_val_bin == 0) & (y_pred_bin == 1)))
        fn = int(np.sum((y_val_bin == 1) & (y_pred_bin == 0)))
        precisao = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precisao * recall / (precisao + recall) if (precisao + recall) > 0 else 0.0

        registros.append({**hp, "f1_binario_inner_val": f1, "tp": tp, "fp": fp, "fn": fn})
        if f1 > melhor_f1:
            melhor_f1 = f1
            melhor_hp = hp

    return melhor_hp, pd.DataFrame(registros)


def executar_cenario(
    familia: str,
    recorte: str,
    protocolo: str,
    regime: str,
    *,
    dir_proc: Path = DIR_DADOS_PROCESSADO,
    dir_artefatos: Path = DIR_ARTEFATOS_EXPERIMENTOS,
    dir_prompts: Path = DIR_ARTEFATOS_PROMPTS,
    hp_de: Path | None = None,
    subset_n: int | None = None,
    subset_seed: int = 2026,
) -> Path:
    """
    Executa um cenário end-to-end. Retorna o diretório do experimento.

    `subset_n`, quando dado, reduz o corpus para smoke validation
    preservando prevalência por mês (amostragem estratificada por mês).
    """
    caminho_corpus = _caminho_corpus(recorte, dir_proc)
    if not caminho_corpus.exists():
        raise FileNotFoundError(
            f"Parquet do recorte ausente: {caminho_corpus}. "
            f"Rode `scripts/preprocessar.py` primeiro."
        )

    df = pd.read_parquet(caminho_corpus)
    if subset_n is not None and subset_n < len(df):
        df = _subamostrar_por_mes(df, subset_n, seed=subset_seed)

    if regime == "bin":
        df = _projetar_df_binario(df)

    classes_treino = sorted(df["y_colapsado"].unique().tolist())

    dir_exp = montar_dir_experimento(familia, recorte, protocolo, regime, raiz=dir_artefatos)
    _configurar_log(dir_exp)
    logging.info(
        "Cenário %s-%s-%s-%s iniciado; corpus=%d artigos; classes=%d",
        familia, recorte, protocolo, regime, len(df), len(classes_treino),
    )

    folds = _gerar_folds(df, protocolo)
    pares_near_dup = _carregar_pares_se_disponivel()
    if pares_near_dup is not None:
        diag_leakage = diagnosticar_leakage(folds, df, pares_near_dup)
        salvar_json(dir_exp / "leakage_diagnostic.json", diag_leakage)

    motor_llama = _iniciar_llama_se_necessario(familia)

    prompt_artifact = _resolver_prompt(familia, recorte, regime, dir_prompts)
    hp_fixo_llama = None
    if familia == "llama31-8b-zs" and prompt_artifact is not None:
        hp_fixo_llama = {"prompt_path": str(prompt_artifact)}

    metricas_por_fold: list[dict] = []
    hp_por_fold: list[dict] = []

    for fold in folds:
        dir_fold = dir_exp / "folds" / f"fold_{fold.indice}"
        dir_fold.mkdir(parents=True, exist_ok=True)

        df_treino = df.iloc[fold.treino_idx]
        df_teste = df.iloc[fold.teste_idx]
        df_inner_train = df.iloc[fold.inner_train_idx]
        df_inner_val = df.iloc[fold.inner_val_idx]

        melhor_hp = _resolver_hp(
            familia=familia,
            fold_idx=fold.indice,
            eh_primario_=eh_primario(recorte, protocolo, regime),
            hp_de=hp_de,
            hp_fixo_llama=hp_fixo_llama,
            df_inner_train=df_inner_train,
            df_inner_val=df_inner_val,
            classes_treino=classes_treino,
            dir_fold=dir_fold,
            motor_llama=motor_llama,
        )
        salvar_json(dir_fold / "best_hp.json", _hp_serializavel(melhor_hp))
        hp_por_fold.append({"fold": fold.indice, "hp": _hp_serializavel(melhor_hp)})

        y_pred, scores, classes_saida, extras = _treinar_e_predizer(
            familia=familia,
            df_treino=df_treino,
            df_teste=df_teste,
            hp=melhor_hp,
            classes_treino=classes_treino,
            dir_tmp=dir_fold / "tmp",
            motor_llama=motor_llama,
        )
        salvar_predictions(dir_fold, df_teste, y_pred, scores, classes_saida)

        metricas = calcular_metricas_fold(
            y_true=df_teste["y_colapsado"].to_numpy(),
            y_pred=y_pred,
            scores=scores,
            classes=classes_saida,
            y_original=df_teste["y_original"].to_numpy(),
        )
        if extras:
            metricas.update(extras)
        salvar_json(dir_fold / "metrics.json", metricas)
        metricas_por_fold.append(metricas)

        logging.info(
            "fold %d: F1bin=%.4f PR-AUCbin=%s macroF1=%.4f n_teste=%d",
            fold.indice,
            metricas["f1_binario_projetado"],
            (
                f"{metricas['pr_auc_binario_projetado']:.4f}"
                if metricas["pr_auc_binario_projetado"] is not None
                else "NA"
            ),
            metricas["macro_f1"],
            metricas["n_teste"],
        )

    resumo = resumir_cross_fold(metricas_por_fold)
    resumo["por_fold"] = [
        {
            "fold": i,
            "f1_binario_projetado": m["f1_binario_projetado"],
            "pr_auc_binario_projetado": m["pr_auc_binario_projetado"],
            "macro_f1": m["macro_f1"],
            "n_teste": m["n_teste"],
        }
        for i, m in enumerate(metricas_por_fold)
    ]
    salvar_json(dir_exp / "summary.json", resumo)

    metadata = construir_metadata(
        familia=familia,
        recorte=recorte,
        protocolo=protocolo,
        regime=regime,
        n_classes_treino=len(classes_treino),
        n_folds=len(folds),
        hp_search_space={"grid": _grid_familia(familia)},
        hp_por_fold=hp_por_fold,
        corpus_raw=ARQUIVO_CSV_RAW,
        corpus_processado=caminho_corpus,
        n_artigos=len(df),
        prompt_artifact=prompt_artifact,
        modo_uso="inference-only" if familia == "llama31-8b-zs" else "train",
    )
    salvar_json(dir_exp / "metadata.json", metadata)

    logging.info("Cenário concluído em %s", dir_exp)
    return dir_exp


def _carregar_pares_se_disponivel() -> pd.DataFrame | None:
    if not CAMINHO_PARES_NEAR_DUP.exists():
        logging.warning("CSV de near-dups ausente em %s", CAMINHO_PARES_NEAR_DUP)
        return None
    return carregar_pares_near_dup(CAMINHO_PARES_NEAR_DUP)


def _iniciar_llama_se_necessario(familia: str):
    if familia != "llama31-8b-zs":
        return None
    from src.modelos.llama_zero import carregar_motor_vllm

    return carregar_motor_vllm()


def _resolver_prompt(
    familia: str, recorte: str, regime: str, dir_prompts: Path
) -> Path | None:
    if familia != "llama31-8b-zs":
        return None
    nome = PROMPT_POR_CENARIO.get((recorte, regime))
    if nome is None:
        return None
    return dir_prompts / nome


def _resolver_hp(
    *,
    familia: str,
    fold_idx: int,
    eh_primario_: bool,
    hp_de: Path | None,
    hp_fixo_llama: dict | None,
    df_inner_train: pd.DataFrame,
    df_inner_val: pd.DataFrame,
    classes_treino: list[str],
    dir_fold: Path,
    motor_llama,
) -> dict:
    """Aplica a regra (ii) da ADR 009 §D.2."""
    if hp_fixo_llama is not None:
        return hp_fixo_llama

    if eh_primario_:
        melhor_hp, tabela = _buscar_melhor_hp(
            familia=familia,
            df_inner_train=df_inner_train,
            df_inner_val=df_inner_val,
            classes_treino=classes_treino,
            dir_tmp=dir_fold / "hp_tmp",
            motor_llama=motor_llama,
        )
        salvar_hp_search(dir_fold, tabela)
        return melhor_hp

    if hp_de is None:
        raise ValueError(
            "Cenário é ablation (regra ii da ADR 009 §D.2) mas `--hp-de` "
            "não foi fornecido. Passe o caminho do experimento primário "
            "correspondente (ex.: artifacts/experimentos/.../fold_0/best_hp.json)."
        )
    return carregar_best_hp_do_primario(hp_de, fold_idx=0)


def _hp_serializavel(hp: dict) -> dict:
    """JSON não aceita `None` como valor de `class_weight` em alguns
    consumidores externos, mas aqui basta converter não-primitivos para
    string (ex.: Path)."""
    return {k: (str(v) if isinstance(v, Path) else v) for k, v in hp.items()}


def _subamostrar_por_mes(
    df: pd.DataFrame, n: int, seed: int
) -> pd.DataFrame:
    """Amostragem estratificada por mês preservando prevalência temporal."""
    rng = np.random.default_rng(seed)
    df_ordenado = df.sort_values(by=["date", "link"], kind="mergesort").reset_index(drop=True)
    meses = df_ordenado["date"].dt.to_period("M")
    grupos = df_ordenado.groupby(meses, sort=True)
    frac = n / len(df_ordenado)

    partes: list[pd.DataFrame] = []
    for _, grupo in grupos:
        tamanho = max(1, int(round(len(grupo) * frac)))
        idx = rng.choice(grupo.index.to_numpy(), size=min(tamanho, len(grupo)), replace=False)
        partes.append(grupo.loc[sorted(idx)])
    subset = pd.concat(partes).sort_values(by=["date", "link"], kind="mergesort").reset_index(drop=True)
    return subset


def _configurar_log(dir_exp: Path) -> None:
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    # Remove handlers antigos (pytest/interativo deixam vários).
    for h in list(logger.handlers):
        logger.removeHandler(h)

    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    handler_arquivo = logging.FileHandler(dir_exp / "run.log", encoding="utf-8")
    handler_arquivo.setFormatter(formatter)
    handler_stream = logging.StreamHandler()
    handler_stream.setFormatter(formatter)
    logger.addHandler(handler_arquivo)
    logger.addHandler(handler_stream)
