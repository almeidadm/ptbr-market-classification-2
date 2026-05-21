"""
Constantes canônicas do projeto: seed, caminhos, recortes de classes,
thresholds, grids de hiperparâmetros e configuração de pipelines.

Centraliza valores que aparecem em múltiplos módulos (carregamento,
splitting, modelos, runner) para que alterações metodológicas tenham um
único ponto de mudança, rastreável via git.

Referências: ADRs 004 (recortes), 006 (tarefa/métrica), 007 (dedup),
008 (splitting), 009 (setup experimental).
"""
from __future__ import annotations

import os
from pathlib import Path

SEED: int = 2026

RAIZ_PROJETO: Path = Path(__file__).resolve().parent.parent

DIR_DADOS_RAW: Path = Path(
    os.environ.get("PTBR_MC_DIR_DADOS_RAW", RAIZ_PROJETO / "data" / "raw")
)
DIR_DADOS_PROCESSADO: Path = Path(
    os.environ.get("PTBR_MC_DIR_DADOS_PROC", RAIZ_PROJETO / "data" / "processado")
)
DIR_ARTEFATOS_EDA: Path = Path(
    os.environ.get("PTBR_MC_DIR_EDA", RAIZ_PROJETO / "artifacts" / "eda")
)
DIR_ARTEFATOS_EXPERIMENTOS: Path = Path(
    os.environ.get("PTBR_MC_DIR_EXP", RAIZ_PROJETO / "artifacts" / "experimentos")
)
DIR_ARTEFATOS_PROMPTS: Path = RAIZ_PROJETO / "artifacts" / "prompts"

ARQUIVO_CSV_RAW: Path = DIR_DADOS_RAW / "articles.csv"

# Classe positiva da tarefa binária projetada (ADR 006).
CLASSE_POSITIVA: str = "mercado"

# Recorte primário da ADR 004: 8 classes (top-6 concorrentes + mercado + outros).
CLASSES_OPCAO_7: tuple[str, ...] = (
    "poder",
    "colunas",
    "mercado",
    "esporte",
    "mundo",
    "cotidiano",
    "ilustrada",
    "outros",
)

# Ablation Opção 4 (21 classes): enumeração depende de recortar o corpus
# pós-dedup com THRESHOLD_OPCAO_4=500, removendo `colunas` e `ilustrada`.
# `scripts/preprocessar.py --recortes opcao4` materializa a lista em
# `data/processado/enumeracao_opcao4.json` antes do consumo pelo runner.
CLASSES_OPCAO_4: tuple[str, ...] | None = None

# Ablation Opção 3 (Garcia-style, 5 classes). Confirmada pelo usuário
# no ciclo 2026-04-24; registrar formalmente em revisão da ADR 004.
CLASSES_OPCAO_3: tuple[str, ...] = (
    "poder",
    "mercado",
    "esporte",
    "mundo",
    "cotidiano",
)

# Thresholds de minor-class rollup (ADR 004).
THRESHOLD_OPCAO_7: int = 1000
THRESHOLD_OPCAO_4: int = 500

# Enumerações canônicas. Ordem fixa → logs e paths estáveis.
RECORTES: tuple[str, ...] = ("opcao7", "opcao4", "opcao3")
FAMILIAS: tuple[str, ...] = ("logreg", "svm", "bertimbau", "llama31-8b-zs")
PROTOCOLOS: tuple[str, ...] = ("rolling", "kfold")
REGIMES: tuple[str, ...] = ("multi", "bin")

# Entrada do classificador (Q1 do ciclo 2026-04-24): junção título+corpo
# uniforme para todas as famílias. Sem separador explícito: concatenação
# com espaço único é suficiente para preservar fronteira de tokens em
# TF-IDF e tokenizers de BERTimbau/Llama.

# Truncamento por família (Q2 do ciclo 2026-04-24).
MAX_PALAVRAS_LLAMA: int = 1500
MAX_TOKENS_BERTIMBAU: int = 256

# TF-IDF compartilhado por LogReg e SVM (ADR 009 §D.3).
TFIDF_CONFIG: dict = {
    "ngram_range": (1, 2),
    "max_features": 100_000,
    "min_df": 5,
    "sublinear_tf": True,
    "strip_accents": "unicode",
    "lowercase": True,
}

# Grid LogReg: 6 configs (ADR 009 §D.3).
GRID_LOGREG_C: tuple[float, ...] = (0.1, 1.0, 10.0)
GRID_LOGREG_CLASS_WEIGHT: tuple[str | None, ...] = (None, "balanced")
LOGREG_FIXOS: dict = {
    "solver": "liblinear",
    "max_iter": 2000,
    "random_state": SEED,
}

# Grid SVM: 6 configs (ADR 009 §D.3).
# Q8 do ciclo 2026-04-24 (Leitura B): `LinearSVC` puro em ambos os
# protocolos, sem `CalibratedClassifierCV`. PR-AUC calculada sobre
# `decision_function` em lugar de `predict_proba`. Divergência em relação
# à redação original da ADR 009 §D.3 registrada aqui — revisão formal
# da ADR pendente.
GRID_SVM_C: tuple[float, ...] = (0.1, 1.0, 10.0)
GRID_SVM_CLASS_WEIGHT: tuple[str | None, ...] = (None, "balanced")
SVM_FIXOS: dict = {
    "dual": "auto",
    "max_iter": 5000,
    "random_state": SEED,
}

# Grid BERTimbau: 6 configs (ADR 009 §D.3).
GRID_BERTIMBAU_LR: tuple[float, ...] = (2e-5, 3e-5, 5e-5)
GRID_BERTIMBAU_EPOCHS: tuple[int, ...] = (3, 4)
BERTIMBAU_FIXOS: dict = {
    "model_name": "neuralmind/bert-base-portuguese-cased",
    "per_device_train_batch_size": 32,
    "warmup_ratio": 0.1,
    "weight_decay": 0.01,
    "max_seq_length": MAX_TOKENS_BERTIMBAU,
    "optim": "adamw_torch",
    "lr_scheduler_type": "linear",
    "seed": SEED,
}

# Llama 3.1 8B zero-shot (ADR 009 §D.3 + Q3 do ciclo 2026-04-24: GPTQ).
# Nota: a disponibilidade do checkpoint específico no HF será validada
# na Fase 3; fallback documentado é `TheBloke/Llama-3.1-8B-Instruct-GPTQ`.
LLAMA_MODEL_ID: str = "hugging-quants/Meta-Llama-3.1-8B-Instruct-GPTQ-INT4"
LLAMA_QUANTIZATION: str = "gptq"
LLAMA_TEMPERATURE: float = 0.0
LLAMA_TOP_P: float = 1.0
LLAMA_MAX_NEW_TOKENS: int = 32

# Protocolo primário: rolling-origin expanding window (ADR 008).
ROLLING_JANELA_TREINO_INICIAL_MESES: int = 18
ROLLING_JANELA_TESTE_MESES: int = 3
ROLLING_N_FOLDS: int = 5
INNER_VAL_MESES: int = 3

# Ablation: stratified k-fold (ADR 008).
KFOLD_K: int = 5

# Bootstrap para ICs cross-fold (ADR 009 §D.6).
BOOTSTRAP_N: int = 1000

# --- Detecção e caracterização de drift (ADRs 010 e 011) ---

DIR_ARTEFATOS_DRIFT: Path = Path(
    os.environ.get("PTBR_MC_DIR_DRIFT", RAIZ_PROJETO / "artifacts" / "drift")
)

# Granularidades de janela para B1 (estatístico) e B2 (semântico).
# B3 (CPD) usa centróides diários — sem entrada aqui por ser parâmetro
# interno do bloco, não escolha de janelamento externa.
GRANULARIDADES_DRIFT: tuple[str, ...] = ("mensal", "bisemanal")

# Escopos da análise condicional por classe (ADR 010 §D.2).
ESCOPOS_DRIFT: tuple[str, ...] = ("global", "mercado", "nao_mercado")

# Repetições do baseline aleatorizado e nível descritivo de significância
# (ADR 011 §D.5 e §D.4).
N_REPETICOES_DRIFT: int = 5
ALPHA_DRIFT: float = 0.05

# Testes do bloco B1 (ADR 010 §D.1.1).
NOMES_TESTES_B1: tuple[str, ...] = ("KS", "CVM", "KTS", "LSDD")

# Tamanho do passo da janela bi-semanal (ADR 011 §D.2).
DIAS_POR_JANELA_BISEMANAL: int = 14

# Cobertura temporal efetiva para drift (ADR 003; revisão R2 da ADR 011).
# Out/2017 (n=121) é excluído da iteração de janelas e do baseline
# aleatorizado para resolver a assimetria documentada na R2 — janelas
# parciais com volume baixíssimo (`mercado` com apenas 16 artigos)
# tornam KTS/LSDD frágeis e contaminam o baseline randomized se ficarem
# espalhados nas pseudo-janelas.
DATA_INICIO_DRIFT: str = "2015-01-01"
DATA_FIM_DRIFT_EXCLUSIVO: str = "2017-10-01"
