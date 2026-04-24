# Notebooks Colab

Adaptadores finos para executar a pipeline no Google Colab. **Nenhuma lógica científica vive aqui** — toda lógica está em `src/` e `scripts/` (CLAUDE.md §5).

## Pré-requisitos

- Cópia do `articles.csv` no Google Drive, em `MyDrive/ptbr-market-classification/data/raw/` (ajustável no notebook).
- URL do seu fork do repositório — editar `REPO_URL` na seção "Parâmetros".
- Runtime → Change runtime type:
  - CPU basta para `preprocessar.ipynb`.
  - T4 cobre LogReg / SVM / BERTimbau.
  - **L4 obrigatória** para Llama 3.1 8B zero-shot (ciclo 2026-04-24 Q4).

## Ordem de execução (ADR 009 §D.7)

1. **`preprocessar.ipynb`** — uma única vez. Gera `corpus_opcao{7,4,3}.parquet` e `hashes.json` em `data/processado/` (persistidos no Drive).
2. **`experimentar.ipynb`** — uma corrida por cenário. Ordem recomendada:
   1. `logreg opcao7 rolling multi` (primário)
   2. `logreg` ablations (`opcao4`, `opcao3`, `bin`, `kfold`) → passar `HP_DE` apontando para o diretório do primário
   3. `svm` primário + ablations
   4. `bertimbau` primário (T4/L4) + ablations
   5. `llama31-8b-zs` todos os cenários (L4) — sem HP search, `HP_DE` fica `None`

Artefatos são persistidos em `MyDrive/ptbr-market-classification/artifacts/experimentos/`, um diretório por corrida seguindo o contrato da ADR 009 §D.6.

## Solução de problemas

- **`git pull` falha com conflito**: no Colab, o clone é volátil. Remova o diretório (`!rm -rf /content/ptbr-market-classification`) e rode a célula de clone novamente.
- **`pip install` lento**: vLLM e torch somam ~3-5 minutos. Normal em runtime novo.
- **BERTimbau estoura VRAM**: T4 é borderline em sequências longas. Subir runtime para L4 ou reduzir `per_device_train_batch_size` via edição temporária de `src/config.py`.
- **Llama off-label alto (>5%)**: revisar prompt do recorte em `artifacts/prompts/` e incrementar versão (`v2`) antes de re-executar — nunca sobrescrever (ADR 009 §D.4).
