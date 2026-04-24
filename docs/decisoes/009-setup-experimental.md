---
numero: 009
slug: setup-experimental
data: 2026-04-24
ciclo: pré-experimentação
status: aceita
---

# 009 — *Setup* experimental: famílias de modelo, protocolo de *tuning*, *grid* de hiperparâmetros, contrato de artefatos

## Contexto

Todas as decisões metodológicas anteriores convergem aqui. Esta ADR
materializa **o que é executado**, **com quais hiperparâmetros**, **que
artefatos são produzidos**, e **em que ordem**. Restrições herdadas:

1. **Recortes do corpus** (ADR 004): primário Opção 7 (8 classes);
   ablations Opção 4 (21 classes) e Opção 3 (5 classes).
2. **Regime de treino e métrica primária** (ADR 006): multiclasse no
   treino; F1 e PR-AUC de `mercado` via projeção binária do argmax;
   ablation binária pura como item 4 da §Decisão da ADR 006.
3. **Política de deduplicação** (ADR 007): corpus pós-dedup (~160.218
   artigos) com coluna `eh_duplicata_text`; `y_original` preservada em
   todos os datasets.
4. **Protocolo de *splitting*** (ADR 008): *rolling-origin expanding
   window* com 5 folds × janela inicial 18 meses × teste 3 meses no
   primário; *stratified k-fold* k=5 sem *grouping* na ablation; *inner
   temporal split* para tuning; *seed* global `2026`.

O ciclo de decisão que produziu esta ADR explorou quatro eixos (famílias
de modelo; orçamento de *tuning* por ablation; *grid* de hiperparâmetros;
contrato de artefatos), mais três subquestões específicas ao uso de LLM
aberta (modelo, modo, *prompt*). As respostas do aluno foram fornecidas
em duas rodadas (2026-04-24) e estão consolidadas na §Decisão abaixo.

## Alternativas consideradas

### Eixo A — Famílias de modelo

| Opção | Famílias | Custo | Comentário |
|---|---|---|---|
| A1 | TF-IDF+LogReg, BERTimbau | baixo-médio | mínimo defensável |
| A2 | TF-IDF+LogReg, TF-IDF+SVM, BERTimbau | médio | 3 famílias, contraste linear × contextual |
| **A3** *(adotada)* | TF-IDF+LogReg, TF-IDF+SVM, BERTimbau, Llama 3.1 8B zero-shot | médio-alto | 4 famílias, inclui paradigma *frozen* LLM |
| A4 | A3 + Word2Vec/fastText + BERTimbau *large* | alto | *benchmark* de volume, pouco contraste novo |

### Eixo B — Orçamento de *tuning* em ablations

| Regra | Custo relativo | Decisão |
|---|---|---|
| (i) *Tuning* por fold em TUDO | 1× (referência) | rejeitada (cost-benefit ruim) |
| **(ii)** *Tuning* por fold só no primário; ablations reusam HPs do fold 0 do primário | ~0,25× | **adotada** |
| (iii) *Tuning* único (fold 0) em tudo | ~0,2× | rejeitada (viola ADR 008 no primário) |
| (iv) *Tuning* por fold no primário; *tuning* único nas ablations | ~0,3× | rejeitada em favor de (ii) |

### Eixo C — *Grid* de hiperparâmetros por família

| Opção | Tamanho | Comentário |
|---|---|---|
| C1 | 3 *configs* | enxuto, pouco defensável |
| **C2** *(adotada)* | 6 *configs* | meio-termo, padrão BRACIS |
| C3 | 12-16 *configs* | rigoroso, custo elevado |

### Eixo D — Persistência de artefatos de modelo

| Opção | Custo armazenamento | Reprodutibilidade |
|---|---|---|
| D1 | salvar todos os modelos de todos os folds | muito alto (>100 GB BERTimbau) | forte |
| D2 | salvar só fold 0 do primário por família | médio (~2 GB) | média |
| **D3** *(adotada)* | salvar só metadados + predições + métricas | baixo (<1 GB) | fraca (reproduzível via re-treino determinístico) |

### Eixos específicos ao LLM

- **Modelo.** Candidatos: Sabiá-3, Llama 3.1 8B, Gemma 2 9B, Qwen 2.5
  7B, Mistral 7B. **Adotado: Llama 3.1 8B** (ubíquo em 2024-25, licença
  compatível, PT-BR forte em *multilingual*).
- **Modo de uso.** Candidatos: zero-shot, few-shot, LoRA *fine-tuning*,
  *embeddings*+classificador. **Adotado: apenas zero-shot**.
- **Cobertura experimental.** Candidatos: só primário (α) ou em todas as
  ablations (β). **Adotado: (β) em todas as ablations**.

## Decisão

### D.1 Famílias de modelo

Quatro famílias no escopo do artigo:

1. **TF-IDF + Regressão Logística** (`sklearn.linear_model.LogisticRegression`).
2. **TF-IDF + SVM linear** (`sklearn.svm.LinearSVC` envolto em
   `CalibratedClassifierCV` para `predict_proba` nativo, necessário para
   PR-AUC).
3. **BERTimbau base** *fine-tuned* — `neuralmind/bert-base-portuguese-cased`.
4. **Llama 3.1 8B zero-shot** — `meta-llama/Llama-3.1-8B-Instruct`, em
   *inference-only*.

ZeroBERTo, BERTimbau *large*, Word2Vec, fastText, LLMs via API pagas e
LoRA *fine-tuning* de LLM ficam fora do escopo.

### D.2 Protocolo de *tuning* — regra (ii)

- **Primário (Opção 7 + *rolling-origin*, multiclasse):** busca de
  hiperparâmetros re-executada integralmente a cada um dos 5 folds via
  *inner temporal split* (ADR 008).
- **Ablations** (Opção 4; Opção 3; binária pura da ADR 006; *stratified
  k-fold* da ADR 008): **reusam os hiperparâmetros selecionados no fold
  0 do primário**, fixos em todos os folds da ablation. Sem busca nova.
- **Llama zero-shot:** sem *tuning* (1 *config* fixa = *prompt*
  registrado em artefato versionado).
- **Critério de seleção dentro do *inner val*:** F1 de `mercado` binária
  projetada (métrica primária ADR 006).

### D.3 *Grid* de hiperparâmetros

**TF-IDF (fixo para LogReg e SVM):**
- `ngram_range=(1, 2)`
- `max_features=100_000`
- `min_df=5`
- `sublinear_tf=True`
- `strip_accents="unicode"`
- `lowercase=True`

**Regressão Logística — *grid* de 6:**
- `C ∈ {0.1, 1.0, 10.0}` × `class_weight ∈ {None, "balanced"}`
- Fixos: `solver="liblinear"` (multiclasse via one-vs-rest), `max_iter=2000`,
  `random_state=2026`.

**SVM linear — *grid* de 6:**
- `C ∈ {0.1, 1.0, 10.0}` × `class_weight ∈ {None, "balanced"}`
- Fixos: `LinearSVC(dual="auto", max_iter=5000, random_state=2026)`,
  envolto em `CalibratedClassifierCV(method="sigmoid", cv=3)`.

**BERTimbau base — *grid* de 6:**
- `learning_rate ∈ {2e-5, 3e-5, 5e-5}` × `num_train_epochs ∈ {3, 4}`
- Fixos: `per_device_train_batch_size=32`, `warmup_ratio=0.1`,
  `weight_decay=0.01`, `max_seq_length=256`, `optimizer="adamw_torch"`,
  `lr_scheduler="linear"`, `seed=2026`.
- *Tokenização:* truncamento à direita em 256 *tokens* após concatenação
  de `title` + separador + `text`.

**Llama 3.1 8B zero-shot — sem *grid*:**
- 1 *config* por regime (multiclasse por recorte; binário na ablation
  da ADR 006).
- `temperature=0`, `top_p=1.0`, `do_sample=False`, greedy.
- `max_new_tokens=32`.
- Quantização 4-bit (AWQ ou GPTQ) para caber em GPU Colab T4/L4.
- *Framework* de inferência: **vLLM** (batelamento e *paging* nativos).

### D.4 *Prompt* do Llama

Um *prompt* fixo por recorte (não um só): a estrutura e as instruções são
idênticas; o que muda é a lista de classes e respectivas descrições. Cada
variante é registrada como artefato próprio em `artifacts/prompts/` e
*hasheada* no `metadata.json` do experimento que a consumir.

Artefatos a produzir:

- `artifacts/prompts/mercado-classification-zero-opcao7-v1.md` — 8 classes.
- `artifacts/prompts/mercado-classification-zero-opcao4-v1.md` — 21 classes.
- `artifacts/prompts/mercado-classification-zero-opcao3-v1.md` — 5 classes.
- `artifacts/prompts/mercado-classification-zero-binario-v1.md` — 2 classes.

**Contrato mínimo de cada *prompt*:**
1. Instrução em PT-BR identificando a tarefa (*"classifique o texto
   abaixo em uma das categorias"*).
2. Lista de categorias com descrição curta (1-2 linhas cada).
3. Instrução de formato de saída canônica (*"responda APENAS o nome
   da categoria, sem explicações"*).
4. Delimitadores claros entre instrução, texto de entrada e resposta.

**Pós-processamento da resposta:**
- *Parse* por *regex* do primeiro termo alfabético da saída do modelo
  (*case-insensitive*, removendo acentos na comparação).
- Correspondência exata contra a lista de classes do recorte.
- *Fallback* em caso de resposta fora das categorias esperadas:
  classificar como a classe **mais frequente do treino do fold**
  (minimiza viés sistemático em favor de `outros` ou `mercado`).
- Reportar, por fold, a **taxa de respostas *off-label***
  (`n_fallback / n_test`) como diagnóstico em `metrics.json`.

**Responsabilidade de redação:** rascunho inicial redigido pelo agente,
revisado e aprovado pelo aluno antes do primeiro uso. Qualquer revisão
posterior do texto exige *bump* de versão (v2, v3…) com artefato novo —
o anterior não é sobrescrito.

### D.5 Cobertura experimental completa

Por família e por modo de *split*, os experimentos executados são:

| Família | Opção 7 *rolling* multi | Opção 7 *rolling* bin | Opção 4 *rolling* multi | Opção 3 *rolling* multi | *Stratified k-fold* multi (Opção 7) |
|---|---|---|---|---|---|
| LogReg | sim (primário) | sim (ablation) | sim (ablation) | sim (ablation) | sim (ablation) |
| SVM | sim (primário) | sim (ablation) | sim (ablation) | sim (ablation) | sim (ablation) |
| BERTimbau | sim (primário) | sim (ablation) | sim (ablation) | sim (ablation) | sim (ablation) |
| Llama zero-shot | sim | sim | sim | sim | sim |

Total: **5 cenários × 4 famílias = 20 "corridas de cenário"**, onde cada
corrida envolve 5 folds. Para as 3 famílias treinadas, apenas a coluna
*primário* executa a busca completa de hiperparâmetros (6 *configs* × 5
folds = 30 *trainings*); as demais colunas reusam HPs e executam 1
*config* × 5 folds = 5 *trainings* cada. Llama não executa *training* em
nenhuma célula — apenas inferência.

Contagem de *trainings* por família treinada:
- LogReg: 30 (primário) + 4 × 5 (ablations) = **50 *fits***. Trivial.
- SVM: 30 + 20 = **50 *fits***. Minutos em CPU.
- BERTimbau: 30 + 20 = **50 *fine-tunings***. Ordem de dias em Colab T4.

Contagem de inferências Llama (test-set cumulativo):
- Opção 7 *rolling*: ~70k artigos (5 folds de teste × ~14k).
- Opção 4 *rolling*: ~50k.
- Opção 3 *rolling*: ~40k.
- Binária (mesmo corpus Opção 7): ~70k.
- *Stratified k-fold* (Opção 7 completo): ~160k.
- Total: **~390k inferências**. A ~15 artigos/s com vLLM 4-bit em L4:
  ~7 horas corridas.

### D.6 Contrato de artefatos

**Estrutura de diretórios:**

```
artifacts/experimentos/
└── <YYYYMMDD-HHMM>-<familia>-<recorte>-<protocolo>-<regime>/
    ├── metadata.json
    ├── folds/
    │   ├── fold_0/
    │   │   ├── hp_search.csv
    │   │   ├── best_hp.json
    │   │   ├── predictions.parquet
    │   │   └── metrics.json
    │   └── fold_1..4/   # mesma estrutura
    ├── summary.json
    ├── leakage_diagnostic.json
    └── run.log
```

Onde:
- `<familia>` ∈ {`logreg`, `svm`, `bertimbau`, `llama31-8b-zs`}.
- `<recorte>` ∈ {`opcao7`, `opcao4`, `opcao3`}.
- `<protocolo>` ∈ {`rolling`, `kfold`}.
- `<regime>` ∈ {`multi`, `bin`}.
- Re-execuções: sufixo `-v2`, `-v3`… no nome do diretório; artefato
  anterior não é sobrescrito.

**Campos mínimos de `metadata.json`:**

```
{
  "experiment_id": "<slug>",
  "timestamp_iso": "<ISO 8601>",
  "git_commit": "<hash>",
  "adr_versions": ["001", "002", ..., "009"],
  "seed": 2026,
  "familia_modelo": "<family>",
  "modo_uso": "<train|inference-only>",
  "recorte": "<opcao7|opcao4|opcao3>",
  "protocolo_split": "<rolling|kfold>",
  "regime": "<multi|bin>",
  "n_classes_treino": <int>,
  "n_folds": 5,
  "janela_inicial_meses": 18,
  "janela_teste_meses": 3,
  "corpus": {
    "raw_sha256": "<hash>",
    "pos_dedup_sha256": "<hash>",
    "n_artigos": <int>
  },
  "hp_search_space": { ... },
  "hp_per_fold": [ {fold: 0, ...}, ... ],
  "prompt_artifact": "<caminho>" ou null,
  "prompt_sha256": "<hash>" ou null,
  "framework_versions": {
    "python": "...",
    "scikit-learn": "...",
    "transformers": "...",
    "torch": "...",
    "vllm": "..."
  }
}
```

**`predictions.parquet` (por fold):**
- Uma linha por artigo no *test set* do fold.
- Colunas mínimas: `link` (id), `date`, `y_original`, `y_colapsado`,
  `y_pred`, `prob_<classe>` para cada classe, `eh_duplicata_text`.

**`metrics.json` (por fold):**
- F1 binária projetada de `mercado` (primária).
- PR-AUC binária projetada de `mercado`.
- Macro-F1 multiclasse.
- Matriz de confusão (`n_classes × n_classes`).
- Composição de FPs de `mercado` por `y_original` (análise A1 da ADR 004).
- Taxa de respostas *off-label* (só para Llama).

**`summary.json`:** agregação cross-fold com média, desvio-padrão e IC
95% via *bootstrap* (1000 re-amostragens com *seed* `2026`).

**`leakage_diagnostic.json`:** fração de pares *near-dup* (ADR 007)
cruzando folds no protocolo atual.

**Persistência:** nenhum *artifact* binário do modelo é salvo —
reprodutibilidade é garantida por `git_commit` + `seed` + `corpus.*_sha256`
+ `hp_per_fold`. Um re-treino determinístico deve reproduzir as
predições dentro de ε numérico razoável.

### D.7 Ordem de execução recomendada

1. **LogReg + Opção 7 *rolling* multi** (primário). Valida pipeline
   end-to-end com custo baixo. Produz HPs de referência do fold 0 para
   consumo nas ablations (regra ii).
2. **LogReg + ablations** (todas as 4). Fecha família rápida antes de
   ir para famílias caras.
3. **SVM + primário + ablations.** Análogo a LogReg.
4. **BERTimbau + primário.** Começa o custo caro; rolling-origin é
   sequencial (fold 1 depende do fold 0 para HPs? não — folds são
   independentes em treino, mas hp_search é per-fold). Pode paralelizar
   folds se múltiplas GPUs disponíveis.
5. **BERTimbau + ablations.** Usa HPs do fold 0 do primário.
6. **Llama zero-shot + todas as 5 colunas.** Inferência pura; pode ser
   executada em paralelo com BERTimbau se GPU separada disponível.

## Justificativa

1. **Quatro famílias cobrem o espectro metodológico.** LogReg (linear
   estatístico com calibração nativa), SVM (linear com kernel e
   calibração via Platt), BERTimbau (contextual profundo *fine-tuned*),
   Llama zero-shot (LLM *frozen*). Cada uma ancora uma pergunta
   diferente sobre o problema. Comparabilidade com Garcia et al. (2024)
   via SVM; com Santana et al. (2022) via BERTimbau.

2. **Regra (ii) de *tuning* balanceia rigor com viabilidade.** O
   primário — onde o rolling-origin é resultado — exige *tuning* por
   fold (ADR 008). As ablations servem para contraste, não para
   produção: reusar HPs do primário mantém o sinal do *gap* entre
   protocolos/recortes legível, evita multiplicar custo em 5× nas
   ablations e é defensável em revisão (*"tuning* foi aplicado no
   cenário que gera resultado principal; ablations operam sob o mesmo
   regime").

3. ***Grid* de 6 *configs* é o padrão defensável em BRACIS.** Suficiente
   para alegar "fizemos busca" sem explodir o orçamento. A escolha de
   dimensões tunadas (`C` e `class_weight` para lineares; `lr` e
   `epochs` para BERTimbau) reflete o que a literatura identifica como
   mais impactante em classificação textual desbalanceada.

4. **Llama 3.1 8B zero-shot em todas as ablations gera perfil de
   deriva *data-side*.** Como o modelo é *frozen*, F1(t) dele mede
   exclusivamente a dificuldade da tarefa por período. Em contraste com
   o F1(t) de BERTimbau (deriva conjunta modelo+dados), dá um
   diagnóstico desacoplado. Contribuição metodológica genuína para o
   eixo "diagnóstico" do artigo (ADR 004).

5. **Persistência apenas de metadados** é coerente com o princípio
   Colab-first: armazenar >100 GB de *checkpoints* no Google Drive é
   inviável e a reprodutibilidade é suficientemente garantida pela
   tripleta (*seed*, *git commit*, *corpus hash*).

6. **Ordem de execução prioriza validação de *pipeline* antes de custo
   alto.** Começar por LogReg detecta *bugs* no código de *splitting*,
   *scoring*, *artifact production* sem queimar horas de BERTimbau.

Contrapontos registrados para consciência (reconhecidos e aceitos):

- ***Stratified k-fold* em Opção 7 inflaciona o volume de inferência
  Llama** (~160k adicionais em um dos cenários, sozinho equivalendo ao
  resto somado). Alternativa seria limitar Llama ao primário e ablations
  rolling-origin; rejeitada porque o *stratified k-fold* é *a* ablation
  que quantifica leakage e otimismo — Llama aplicado a ele mede um
  efeito ortogonal (dificuldade do teste com distribuição alterada), e
  a diferença de custo é absorvível (~3 horas extras).
- **Persistência-fraca torna auditoria externa mais difícil.** Revisor
  BRACIS que questione um número específico não pode inspecionar o
  modelo. Mitigação: `predictions.parquet` retém predições ponto a
  ponto; métricas derivadas são todas reproduzíveis a partir desse
  arquivo sem re-treinar.
- **Pós-processamento com *fallback* para classe mais frequente** pode
  artificialmente inflar F1 da classe majoritária (tipicamente `outros`
  em Opção 7). Reportar a taxa *off-label* por fold é essencial; se
  essa taxa for >5% em qualquer cenário, discutir implicações na
  interpretação da métrica primária do Llama.

## Consequências

### Habilita

- **Início imediato da implementação da *pipeline*** — todas as
  decisões metodológicas que afetam código estão consolidadas.
- **Reprodutibilidade fraca garantida** via tripleta (*seed*, *git
  commit*, *corpus hash*) + artefatos `predictions.parquet`.
- **Comparabilidade direta com Garcia et al. (2024)** via ablation
  Opção 3 + SVM.
- **Eixo Green AI viabilizado** — reporte de FLOPs, tempo de
  *training/inference* por família já cabe no contrato
  `metadata.json` + `run.log`.
- **Narrativa *catch-all diagnóstico*** (ADR 004) operacionalizada —
  `predictions.parquet` com `y_original` permite computar análises A1,
  A2, A3 da ADR 004 diretamente em *post-processing*.

### Impede ou restringe

- **Auditoria de modelo binário não é possível** sem re-treino.
- **Novas famílias de modelo** exigem re-abertura do ciclo
  *setup* — por exemplo, adicionar BERTimbau *large*, Sabiá-3 ou LoRA
  em Llama depende de revisão desta ADR.
- **Modificação do *grid*** de hiperparâmetros invalida resultados
  anteriores — a `hp_search_space` é parte do contrato.
- **Re-execuções parciais** (ex.: só recomputar métricas) exigem disciplina
  de versionamento — o sufixo `-v2`, `-v3` é obrigatório para não
  sobrescrever.

### Dependências com outras ADRs

- **Consome tudo de ADRs 004, 006, 007, 008.**
- **Não altera ADR 005** (WikiNews OOD, em `proposta`, postergada);
  quando for reabrida, segue este contrato de artefatos com adaptações.
- **Precede** o ciclo de **implementação da pipeline** — próximo
  ciclo natural após esta ADR.

### Revisão futura esperada

- **Se Llama zero-shot tiver taxa *off-label* alta** (>5%), revisitar
  política de *fallback* e/ou redesenhar *prompt*.
- **Se BERTimbau estourar orçamento Colab**, cair *grid* de 6 para 4
  *configs* e registrar revisão.
- **Se rolling-origin produzir perfil F1(t) plano** (sem deriva
  perceptível em nenhuma família), revisitar hipótese de trabalho: a
  deriva composicional documentada na ADR 003 não se traduz
  necessariamente em deriva de desempenho.

### Pendências de implementação

- **Módulos a criar:**
  - `src/splitting/rolling_origin.py` (geração de folds temporais).
  - `src/splitting/kfold_stratified.py` (ablation).
  - `src/splitting/leakage.py` (diagnóstico de *near-dup crossings*).
  - `src/modelos/logreg.py`, `src/modelos/svm.py`, `src/modelos/bertimbau.py`,
    `src/modelos/llama_zero.py`.
  - `src/experimento/runner.py` (orquestrador do *fit/eval/persist*).
  - `src/experimento/artefatos.py` (escrita/leitura de `metadata.json`,
    `predictions.parquet`, etc.).
- **Artefatos a produzir** (antes do primeiro experimento):
  - 4 arquivos de *prompt* em `artifacts/prompts/` (rascunho redigido
    pelo agente, revisão do aluno).
  - `src/config.py` com `SEED = 2026` e caminhos padrão.
- **Configuração Colab:**
  - Cell padrão de *setup* (mount Drive, clone repo, `pip install -r
    requirements.txt`, `import` do runner).
  - Execução por cenário via argumentos CLI ao `runner`.
- **Testes unitários a escrever:**
  - Determinismo do rolling-origin e *stratified k-fold* (ADR 008).
  - *Parsing* de resposta Llama (respostas válidas, inválidas,
    maiúsculas/minúsculas, acentos).
  - Estrutura de `metadata.json` (obrigatoriedade de campos).
- **Validação de smoke:**
  - Antes da primeira corrida longa, rodar LogReg + Opção 7 rolling
    multi em um subset do corpus (ex.: 1.000 artigos) para verificar
    que toda a pipeline gera artefatos válidos.

## Revisão em 2026-04-24

Alterações posteriores à aceitação original, motivadas pelas respostas
do aluno no ciclo de implementação (Q1–Q8 do mesmo dia) e por
adaptações exigidas pelo código concreto. Nenhum experimento real
havia sido executado antes desta revisão — portanto não há resultados
a invalidar.

### R.1 SVM sem `CalibratedClassifierCV` (Q8, Leitura B)

- **Redação original** (§D.1, §D.3): `LinearSVC` envolto em
  `CalibratedClassifierCV(method="sigmoid", cv=3)` para obter
  `predict_proba` e, daí, PR-AUC.
- **Revisão:** `LinearSVC` **puro** em ambos os protocolos
  (rolling-origin primário e *stratified k-fold* ablation). PR-AUC
  passa a ser calculada sobre `decision_function`.
- **Motivação:** a CV interna aleatória do calibrador (`cv=3`)
  introduziria subsplits aleatórios no *inner train* do rolling-origin,
  quebrando o compromisso de determinismo estrito e assimetria
  temporal herdado da ADR 008. Alternativa `cv="prefit"` sobre o
  *inner val* introduziria pipeline diferente para cada protocolo e
  dificultaria a comparação; Leitura B (pipeline único) escolhida
  pela uniformidade.
- **Consequência no código:** `scores` devolvido por `src.modelos.svm`
  tem escala real (não proba). O cálculo de PR-AUC em
  `src.experimento.metricas` já opera sobre escores reais — bastou
  `precision_recall_curve` aceitar o vetor de `decision_function`
  diretamente.
- **Consequência textual:** o parágrafo "SVM linear — *grid* de 6" de
  §D.3 deve ser lido como: "Fixos: `LinearSVC(dual='auto',
  max_iter=5000, random_state=2026)`" (removida a linha do
  calibrador).

### R.2 Quantização Llama fixada em GPTQ (Q3)

- **Redação original** (§D.3): "Quantização 4-bit (AWQ ou GPTQ) para
  caber em GPU Colab T4/L4."
- **Revisão:** **GPTQ** adotado. Checkpoint canônico
  `hugging-quants/Meta-Llama-3.1-8B-Instruct-GPTQ-INT4` (disponibilidade
  reverificada no primeiro uso; fallback a `TheBloke/*-GPTQ` se
  necessário).
- **Motivação:** em inferência determinística (greedy,
  `temperature=0`) a diferença prática entre AWQ e GPTQ é marginal;
  GPTQ é mais maduro em vLLM via backend Marlin, bem suportado em
  GPUs Ampere+.
- **Infraestrutura (Q4):** **L4 obrigatória** para Llama. T4 (16 GB)
  é insuficiente — KV cache em textos longos (até 1.500 palavras,
  ver R.3) estoura.
- **Materialização:** `LLAMA_MODEL_ID` e `LLAMA_QUANTIZATION` em
  `src/config.py`; `carregar_motor_vllm` em
  `src/modelos/llama_zero.py`.

### R.3 Texto de entrada e truncamento (Q1, Q2)

Ausente na redação original; consolidado nesta revisão para servir de
ponto único de verdade.

- **Q1 — concatenação:** `title + " " + text` com um único espaço
  separador. Sem `\n`, sem `[SEP]`, sem marcadores. Uniforme em todas
  as famílias; diferenças entre modelos ficam isoladas no tokenizador
  de cada um.
- **Q2 — truncamento:** BERTimbau em 256 *tokens* (já em §D.3);
  Llama em **1.500 palavras** (~2k *tokens*, margem para *prompt*
  sobrevier com até 2.048 de contexto em sampling); TF-IDF sem
  truncamento.
- **Q5 — fallback Llama off-label:** classe **mais frequente no
  treino do fold atual** (não no treino global). Mantém a decisão
  registrada em §D.4 mas formaliza o escopo "do fold" para evitar
  viés sistemático dependente do recorte.
- **Materialização:** `src/modelos/texto.py` (`juntar_titulo_texto`,
  `preparar_entrada`) e constantes `MAX_PALAVRAS_LLAMA`,
  `MAX_TOKENS_BERTIMBAU` em `src/config.py`.

### R.4 Estado dos artefatos de *prompt* (§D.4)

Todos os quatro artefatos foram aprovados pelo aluno em 2026-04-24 e
estão prontos para consumo em corrida real. Frontmatter YAML de cada
arquivo carrega `status: aceita` e `aprovado_em: 2026-04-24`.

- `mercado-classification-zero-opcao7-v1.md` — 8 classes.
- `mercado-classification-zero-opcao3-v1.md` — 5 classes
  `{poder, mercado, esporte, mundo, cotidiano}` (pendência da ADR
  004 §D.5 resolvida).
- `mercado-classification-zero-binario-v1.md` — 2 classes
  `{mercado, nao-mercado}`.
- `mercado-classification-zero-opcao4-v1.md` — 21 classes
  enumeradas empiricamente por `scripts/preprocessar.py` sobre o
  corpus pós-dedup (ADR 007). As nove não listadas na primeira
  pendência da ADR 004 §D.5 (`ciencia`, `comida`, `educacao`,
  `empreendedorsocial`, `equilibrioesaude`, `folhinha`,
  `ilustrissima`, `sobretudo`, `turismo`) receberam descrições
  geradas por *template* e aprovadas sem edição (campo
  `descricoes_por_template: true` preservado para auditoria).

Qualquer revisão futura do texto destes artefatos exige *bump* de
versão (v2, v3…) — o v1 aprovado não é sobrescrito (ADR 009 §D.4).

### R.5 Adições de pipeline não previstas em §D

- **`id_raw`** em `src/preprocessamento/carregamento.py`: posição no
  corpus bruto pós-ordenação estável, preservada em todos os
  recortes. Essencial para que `src/splitting/leakage.py` consuma o
  CSV `artifacts/eda/tabelas/07-near-duplicates-pares.csv` produzido
  pela EDA (cujos `id_a`/`id_b` referenciam esse espaço posicional).
- **`--smoke N`** em `scripts/experimentar.py`: subamostragem
  estratificada por mês preservando prevalência temporal. Introduzida
  para validar a pipeline end-to-end antes das corridas completas.
  Subset não substitui nenhuma das 20 corridas canônicas de §D.5.
- **Versionamento automático de diretório de experimento:** sufixo
  `-v2`/`-v3`… aplicado por `montar_dir_experimento` quando um
  diretório com o mesmo timestamp+slug já existe. Complementa a
  regra "re-execuções: sufixo `-v2`..." de §D.6 cobrindo colisões
  dentro do mesmo minuto.

### R.6 Divergências numéricas em relação a contagens pré-pipeline

Ao materializar a Fase 1 (carregamento/dedup/recortes), os tamanhos
efetivos de cada recorte divergiram ligeiramente das estimativas das
ADRs 004 e 007. A análise revelou também um bug de implementação em
`aplicar_opcao_7`, corrigido antes do fechamento desta revisão:

- **Bug encontrado e corrigido:** a primeira versão de
  `src/preprocessamento/recortes.py:aplicar_opcao_7` fazia apenas o
  *relabel* e omitia o passo de `threshold ≥1.000` exigido pela
  §D.5 item 7 da ADR 004 — o que correspondia à Opção 1, não à
  Opção 7. A correção (filtro antes do relabel) foi aplicada
  antes de qualquer corrida real; o único artefato afetado é o
  smoke validation do runner, que permanece válido como
  verificação da pipeline mas não como resultado a reportar.
- **Reconciliação dos "597":** a ADR 007 registra "597 linhas
  removidas"; o pipeline remove **387** linhas e marca **210**
  sobreviventes como `eh_duplicata_text=True`. 387 + 210 = 597
  (linhas *envolvidas* em grupos de duplicação) — número que a
  EDA original contabilizou. Sem conflito.

Tabela final pós-correção:

| Métrica | Estimativa ADR | Pós-pipeline | Delta |
|---|---:|---:|---:|
| Corpus pós-dedup | ~166.456 (ADR 007) | 165.901 | −555 (−0,33%) |
| Opção 7 (threshold + relabel) | 160.815 (ADR 004 §D.5) | 159.681 | −1.134 (−0,71%) |
| Opção 4 | 126.921 (ADR 004 §D.5) | 126.647 | −274 (−0,22%) |
| Opção 3 | 96.819 (ADR 004 §D.5) | 96.742 | −77 (−0,08%) |

As diferenças remanescentes (< 1 %) vêm de dois efeitos não
previstos explicitamente nas estimativas originais: (i) filtro de
linhas com `date`/`text` nulos em `preparar_corpus`, pré-requisito
do protocolo temporal da ADR 008; (ii) ordem de dedup-antes-de-recortes
sobre corpus já filtrado. Nenhuma delas altera conclusões
qualitativas das ADRs 004 ou 007.

**As ADRs 004 e 007 foram atualizadas em 2026-04-24** com seções
próprias de revisão documentando os números definitivos, a correção
do bug da Opção 7, a reconciliação do "597" e a coluna `id_raw`
introduzida para o diagnóstico de leakage.
