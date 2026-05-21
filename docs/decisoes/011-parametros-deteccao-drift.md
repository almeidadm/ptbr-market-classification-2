---
numero: 011
slug: parametros-deteccao-drift
data: 2026-05-21
ciclo: pré-experimentação
status: aceita
depende_de: [008, 009, 010]
---

# 011 — Parâmetros e contrato de artefatos para detecção de drift

## Contexto

A ADR 010 fixou os **blocos metodológicos** (B1 estatístico, B2 semântico,
B3 CPD) e a **análise condicional por classe** (global, `mercado`,
`não-mercado`) como protocolo de caracterização de drift para o
FolhaUOL, com baseline aleatorizado de controle. Os **parâmetros
concretos** (janela, embedding, threshold, volume, repetições) foram
explicitamente deferidos.

Esta ADR consolida esses parâmetros e o contrato de artefatos, fechando
o último ciclo de decisão antes de codificar. Implementação concreta
(módulos, testes, scripts) sai como pendência desta ADR.

Restrições herdadas:

1. **ADR 009 §D.1.3** comprometeu BERTimbau base como família 3 do setup
   experimental. Os embeddings produzidos para inferência são reusados
   aqui sem custo adicional.
2. **ADR 008** fixou granularidade mensal para folds *rolling-origin*.
   Alinhar janelas de drift com isso facilita interpretação cruzada
   entre perfil F1(t) e perfil de drift.
3. **ADR 010 §D.4** registrou WMD como condicional a haver word2vec no
   projeto. A decisão "BERTimbau apenas" desta ADR torna WMD fora de
   escopo *de facto*.
4. **ADR 008 §Decisão** fixou `seed = 2026` como constante do projeto.
   Reusada aqui.

## Alternativas consideradas

### Eixo A — Embedding

| Opção | Descrição | Decisão |
|---|---|---|
| A1 | BERTimbau base apenas | **adotada** |
| A2 | BERTimbau base + word2vec CBOW treinado paralelamente | rejeitada (custo de treino e ausência de demanda específica) |
| A3 | BERTimbau large | rejeitada (sai do escopo da ADR 009; ganho marginal vs custo) |

Consequência direta: WMD removido (incompatível com BERT contextual,
conforme Wanderley §3.2).

### Eixo B — Granularidade de janela

| Bloco | Opção primária | Opção secundária | Diário |
|---|---|---|---|
| **B1 estatístico** | **mensal (33 janelas)** | bi-semanal (66 janelas, anexo/curiosidade) | descartado |
| **B2 semântico** | **mensal (33 janelas)** | bi-semanal (66 janelas, anexo/curiosidade) | descartado |
| **B3 CPD** | **diário (centróides; ~1.005 dias com 18 dias zerados da ADR 003)** | — | — |

A assimetria do B3 é intencional: CPD precisa de granularidade fina para
localizar o ponto de mudança com precisão (Wanderley §3.3). Aplicar CPD
sobre centróides mensais oversmooth-aria a série e esconderia *change
points* relevantes.

### Eixo C — Volume por janela

| Opção | Descrição | Decisão |
|---|---|---|
| C1 | usar todos os artigos da janela (~5k/mês, ~2,5k/bi-semana) | **adotada** |
| C2 | subamostrar para tamanho fixo igual ao mínimo da série | rejeitada (introduz variância sem ganho metodológico) |
| C3 | fixar em 40 como Wanderley | rejeitada (cap do Wanderley era artefato de dataset pequeno; com ~5k/mês não há motivo) |

### Eixo D — Threshold de significância em B1

| Opção | Descrição | Decisão |
|---|---|---|
| D1 | α=0,05 plano como Wanderley | **adotada** |
| D2 | Bonferroni sobre todas as janelas × testes × condições | rejeitada (uso é descritivo, não inferencial estrito) |
| D3 | FDR / Benjamini-Hochberg | rejeitada (mesma razão; sem múltipla rejeição/aceitação como mecânica do achado) |

### Eixo E — Repetições do baseline aleatorizado

| Opção | Decisão |
|---|---|
| E1 — 5 repetições (Wanderley) | **adotada** |
| E2 — 10-20 (mais robusto) | rejeitada (custo cresce linearmente sem ganho narrativo) |
| E3 — 1 (mínimo) | rejeitada (perde IC, perde robustez) |

### Eixo F — Contrato de artefatos

| Opção | Decisão |
|---|---|
| F1 — `metadata.json` + `results.parquet` por execução | **adotada** |
| F2 — Parquet único com metadados embutidos | rejeitada (dificulta inspeção rápida e versionamento de metadados) |
| F3 — CSV plano | rejeitada (sem schema, sem tipos) |

## Decisão

### D.1 Embedding

- **BERTimbau base** (`neuralmind/bert-base-portuguese-cased`, 768-d)
  como único embedding para B1, B2 e B3.
- **Representação por documento**: pooling do `[CLS]` token da última
  camada, conforme prática padrão para classificação. Decisão revisável
  se evidência empírica favorecer *mean-pooling* (TODO de verificação
  rápida durante implementação).
- **Cache de embeddings**: gerar uma vez sobre o corpus pós-dedup
  (~160.218 artigos da ADR 007), persistir em
  `artifacts/embeddings/bertimbau_base_cls/`. Drift detection reusa
  esse cache.
- WMD fora de escopo.

### D.2 Janelas

**B1 (estatístico) e B2 (semântico):**

- **Primário — janelas mensais contíguas.** 33 janelas (jan/2015 a
  set/2017), correspondendo à cobertura efetiva da ADR 003. Out/2017
  (n=121, excluído da agregação mensal pela ADR 003) também excluído
  aqui pela mesma razão.
- **Secundário (curiosidade) — janelas bi-semanais.** ~66 janelas
  contíguas (verificar exato após implementação por causa do gap de
  18 dias da ADR 003). Entra como tabela/figura de anexo, não na
  narrativa principal.

**B3 (CPD):**

- **Centróides diários.** Um centróide por dia do corpus.
- Dias com zero notícias (18 dias de jan-set/2017, ADR 003) são tratados
  como missing: a série de centróides tem ~987 pontos não-nulos.
  Decisão de interpolar ou pular esses pontos durante a aplicação do
  *binary segmentation*: **pular** (não interpolar), porque interpolação
  injeta sinal artificial.
- Redução para série univariada: 1ª componente PCA dos centróides
  diários, conforme Wanderley §4.3.

**Comparações:**

- **Time-ordered**: cada janela contra a janela imediatamente anterior
  (pair-wise consecutivo, como Wanderley Figura 2).
- **Randomized baseline**: para cada repetição do baseline, partições
  do mesmo tamanho da série mensal/bi-semanal, sorteadas **sem
  reposição** sobre o corpus inteiro, quebrando ordem temporal.

### D.3 Volume

- Usar **todos os artigos da janela** sem subamostrar. Para mensal,
  ~5k/janela; para bi-semanal, ~2,5k/janela. Memória de KTS por par
  de janelas no pior caso (mensal vs mensal): ~400 MB float32, viável
  em Colab T4/L4.
- Para análise condicional, "todos" significa todos os de cada classe:
  ~630/mês para `mercado` e ~4,4k/mês para `não-mercado`.

### D.4 Threshold

- **α = 0,05 plano** para B1, sem correção de múltiplas comparações.
- **Uso descritivo, não inferencial.** O reporte do artigo é:
  - Tabela 1 (réplica Wanderley): média e desvio dos p-values agregados
    por teste e por condição (time-ordered vs randomized).
  - Figura central: série temporal dos p-values bi-semanais consecutivos
    com linha horizontal em α=0,05 como referência visual. Conta de
    janelas abaixo do limiar entra como métrica descritiva, não como
    rejeição de hipótese única.

### D.5 Repetições e seeds

- **5 repetições** do baseline aleatorizado.
- **Seed global do projeto**: `2026` (herdada da ADR 008).
- **Seeds derivadas por repetição**: `2026 + repeticao_id` (0 a 4),
  garantindo determinismo e independência entre repetições.
- Métrica reportada: média ± desvio sobre as 5 repetições.

### D.6 Contrato de artefatos

```
artifacts/drift/
├── embeddings/
│   └── bertimbau_base_cls/
│       ├── metadata.json        # modelo, versão, hash do corpus, pooling, seed
│       └── embeddings.parquet   # colunas: article_id, embedding (list[float])
├── b1_statistical/
│   └── {timestamp}_{granularidade}_{escopo}/
│       ├── metadata.json
│       └── results.parquet
├── b2_semantic/
│   └── {timestamp}_{granularidade}_{escopo}/
│       ├── metadata.json
│       └── results.parquet
└── b3_cpd/
    └── {timestamp}_{escopo}/
        ├── metadata.json
        └── results.parquet
```

Onde:
- `{granularidade}` ∈ {`mensal`, `bisemanal`} para B1/B2.
- `{escopo}` ∈ {`global`, `mercado`, `nao_mercado`}.
- `{timestamp}` é ISO-8601 sem caracteres incompatíveis com filesystem
  (formato `YYYYMMDDTHHMMSS`).

**Schema mínimo de `metadata.json`** (a expandir conforme implementação):

```json
{
  "adr_referencia": "011",
  "bloco": "b1_statistical | b2_semantic | b3_cpd",
  "granularidade": "mensal | bisemanal | diario",
  "escopo": "global | mercado | nao_mercado",
  "corpus_hash": "sha256 do CSV de entrada",
  "n_artigos": <int>,
  "embedding": "bertimbau_base_cls",
  "embedding_hash": "sha256 dos vetores",
  "seed_global": 2026,
  "n_repeticoes": 5,
  "alpha": 0.05,
  "tests": ["KS", "CVM", "KTS", "LSDD"],
  "biblioteca_versao": {"alibi-detect": "x.y.z", "ruptures": "x.y.z"},
  "data_execucao": "ISO-8601",
  "duracao_segundos": <float>
}
```

**Schema mínimo de `results.parquet`** (varia por bloco):
- **B1**: colunas `janela_a`, `janela_b`, `teste`, `repeticao`,
  `condicao` (time_ordered|randomized), `p_value`, `estatistica`.
- **B2**: colunas `janela_a`, `janela_b`, `metrica`
  (cosine_centroid|cumulative_cosine|mmd2), `valor`.
- **B3**: colunas `change_point_data`, `posicao_serie`, `escopo`.

### D.7 Estrutura de código

```
src/drift/
├── __init__.py
├── windowing.py      # geração de janelas mensais, bi-semanais, partições aleatorizadas
├── statistical.py    # B1: KS, CVM, KTS, LSDD com agregação Fisher
├── semantic.py       # B2: cosseno centroide, cumulativo, MMD²
├── cpd.py            # B3: binary segmentation via ruptures sobre PCA dos centroides
└── artifacts.py      # leitura/gravação dos metadata.json e results.parquet

scripts/drift/
├── compute_embeddings.py   # produz cache de embeddings BERTimbau
├── run_b1_statistical.py   # executa B1 sobre granularidades × escopos
├── run_b2_semantic.py      # idem para B2
└── run_b3_cpd.py           # idem para B3

tests/drift/
├── test_windowing.py       # determinismo de partições; tamanhos esperados
├── test_statistical.py     # sanidade em sinal sintético com drift conhecido
├── test_semantic.py        # idem
└── test_cpd.py             # idem
```

## Justificativa

1. **Embedding único (BERTimbau base) reduz complexidade combinatória.**
   Cada bloco × granularidade × escopo já gera dezenas de execuções;
   multiplicar por dois embeddings (BERTimbau + word2vec) dobraria
   custo e tabelas sem ganho narrativo claro. A escolha alinha-se com
   ADR 009 sem expandir escopo.

2. **Janela mensal como primário alinha com três peças do desenho.**
   - Mann-Kendall da ADR 003 já é mensal.
   - Rolling-origin da ADR 008 é mensal (folds de 3 meses).
   - Mensal é a granularidade que aparece em toda a narrativa do artigo,
     facilitando leitura cruzada das figuras.

3. **Bi-semanal como curiosidade preserva comparabilidade com Wanderley
   sem alterar narrativa.** Replicar a granularidade do Wanderley
   permite reportar no apêndice "nossos resultados são consistentes
   sob a granularidade do trabalho de referência", o que reforça
   robustez sem exigir reformulação das figuras principais.

4. **CPD diário é mecanicamente necessário.** Aplicar binary
   segmentation sobre 33 pontos mensais daria, no melhor caso, 1-2
   *change points* identificáveis com posicionamento grosseiro. Sobre
   ~987 pontos diários, o algoritmo localiza shifts no nível da
   semana, replicando a interpretabilidade do Wanderley (que apontou
   16/mar/2020 como ponto de início do drift pandêmico).

5. **Usar todos os artigos é livre.** A análise de memória do turno
   anterior mostrou que mensal vs mensal cabe em ~400 MB float32, e
   bi-semanal vs bi-semanal em ~100 MB. Subamostrar introduziria
   variância amostral controlável (já temos a controlada pelo baseline
   randomizado) sem economizar recurso real.

6. **α=0,05 plano é defensável quando o uso é descritivo.** O argumento
   metodológico do artigo não depende de rejeitar uma hipótese única.
   Depende de mostrar que **o perfil temporal de p-values difere
   sistematicamente entre time-ordered e randomized** — comparação que
   é robusta a escolha de α (Wanderley reporta médias de p-values, não
   contagens de rejeição). Correção de múltiplas comparações ficaria
   relevante se o artigo fizesse afirmações pontuais tipo "houve drift
   na janela X com p<0,05 corrigido" — não fazemos isso.

7. **5 repetições é suficiência empírica.** Wanderley reporta desvios
   na ordem de 0,01-0,17 com 5 repetições; aumentar para 10 ou 20
   reduziria o desvio em fator √2 a √4, sem mudar substantivamente a
   diferença observada entre condições (que é da ordem de 0,1-0,3 em
   média de p-values). Custo extra não compensa.

8. **`metadata.json` + `results.parquet` separados** facilitam dois
   fluxos comuns: inspecionar parâmetros sem carregar dados (abrir
   JSON), e cruzar resultados em DataFrame sem reparsear strings.
   Resolve a pendência aberta no §4.2 do CLAUDE.md sobre contrato
   geral de artefatos — embora aqui esteja escopado a drift, o padrão
   é replicável para experimentos de classificação na ADR 009.

9. **`{timestamp}_{granularidade}_{escopo}/` como diretório de execução**
   permite múltiplas execuções coexistirem sem sobrescrita, e
   torna a busca "qual rodada foi essa?" trivial.

Contrapontos registrados:

- **Pooling `[CLS]` vs *mean-pooling*.** Wanderley usa BERTimbau como
  encoder mas não explicita o pooling. Literatura mostra que para
  similaridade semântica frequentemente *mean-pooling* supera `[CLS]`
  (Reimers & Gurevych 2019, Sentence-BERT). Adotamos `[CLS]` por
  alinhamento com o uso de BERTimbau como classificador na ADR 009
  (onde `[CLS]` é entrada da camada de classificação), mantendo
  consistência entre embeddings de drift e embeddings de
  classificação. Se evidência empírica durante implementação favorecer
  *mean-pooling*, reabrir esta decisão.
- **Out/2017 excluído** (ADR 003) significa que set/2017 é a última
  janela mensal; bi-semanal termina em 16-30/set/2017.
- **18 dias zerados em 2017** (ADR 003) afetam:
  - Mensal: marginal — cada mês afetado perde 2 dias de ~30, não muda
    contagem agregada significativamente.
  - Bi-semanal: mais impactante — janela contendo dias 11-12 pode ter
    contagem ~15% menor. Mitigação: documentar no anexo bi-semanal,
    não tentar interpolar.
  - Diário/CPD: os 18 dias entram como missing, tratados como
    pular-não-interpolar.
- **Determinismo de KTS sob subamostragem do baseline aleatorizado.**
  KTS internamente faz um teste de permutação. A seed precisa
  controlar tanto a partição aleatorizada externa quanto a permutação
  interna do teste. Isso vai na implementação como TODO de teste
  unitário.

## Consequências

### Habilita

- **Implementação imediata**. Todas as decisões necessárias para
  codificar `src/drift/` estão tomadas.
- **Comparabilidade direta com Wanderley** nas figuras e tabelas
  (mesmas estatísticas, mesmas convenções de baseline e repetições).
- **Reuso de embeddings entre drift e classificação**. Como o cache de
  BERTimbau `[CLS]` é único, a ADR 009 pode também consumi-lo onde
  fizer sentido (sem refazer *forward pass*).
- **Padrão de contrato de artefatos** estabelecido, herdável pela ADR
  009 e por ciclos futuros.

### Impede ou torna custoso

- **Reabrir escolha de embedding** custa: precisa regerar cache e
  re-executar todos os blocos. Decisão consciente — mudar embedding
  meio-experimento é exatamente o tipo de churn que se quer evitar.
- **Comparar com word2vec** exigiria reabrir ADR 010 D.1 e treinar
  word2vec. Não há demanda atual.
- **Mudar granularidade primária para algo diferente de mensal**
  exigiria realinhar com ADR 008. Custo médio.

### Dependências com outras ADRs

- **Materializa** as escolhas pendentes da ADR 010 §D.4.
- **Reusa** seed `2026` da ADR 008.
- **Reusa** embeddings BERTimbau da ADR 009.
- **Compatibiliza** janelas com cobertura temporal da ADR 003.
- **Não altera** ADRs 001-009.

### Pendências de implementação

1. **Gerar cache de embeddings.**
   - Script `scripts/drift/compute_embeddings.py`.
   - Entrada: CSV do corpus pós-dedup (ADR 007).
   - Saída: `artifacts/drift/embeddings/bertimbau_base_cls/{metadata.json,
     embeddings.parquet}`.
   - Determinismo: `torch.manual_seed(2026)`, `set_seed(2026)`,
     `torch.use_deterministic_algorithms(True)`.

2. **Implementar `src/drift/windowing.py`.**
   - `gerar_janelas_mensais(df) -> List[Tuple[indices, str_label_janela]]`.
   - `gerar_janelas_bisemanais(df) -> List[Tuple[indices, str_label_janela]]`.
   - `gerar_particoes_aleatorizadas(df, n_janelas, tamanho_por_janela,
     seed) -> List[Tuple[indices, str_label_pseudo_janela]]`.
   - Testes: determinismo sob seed fixa; cobertura de calendário sem
     gaps inesperados.

3. **Implementar `src/drift/statistical.py`.**
   - Wrappers sobre `alibi-detect` para KS, CVM, LSDD.
   - Wrapper sobre código Olivetti (ou re-implementação verificada) para
     KTS, com permutation test usando seed derivada.
   - Agregação de p-values via Fisher para KS e CVM (univariados).
   - Testes: sinal sintético com drift conhecido — esperado p-value
     baixo em time-ordered, alto em randomized.

4. **Implementar `src/drift/semantic.py`.**
   - `cosine_centroid_consecutivo(janelas) -> Series`.
   - `cosine_centroid_cumulativo(janelas) -> Series` (contra média
     histórica).
   - `mmd2_consecutivo(janelas, seed) -> Series`.
   - Testes: sinal sintético; igualdade entre duas amostras idênticas.

5. **Implementar `src/drift/cpd.py`.**
   - Função: dado série de centróides diários (com NaN para dias
     zerados), aplicar PCA 1ª componente, depois binary segmentation
     via `ruptures` com kernel gaussiano.
   - Hiperparâmetro de número de breakpoints: usar penalidade BIC
     (default do `ruptures` em `Pelt`) ou número fixo? **Decisão
     adiada para implementação**: testar primeiro com penalidade BIC,
     reabrir se output for ruim.
   - Testes: série sintética com 1 change point conhecido — detector
     deve localizar dentro de ±5% do tamanho da série.

6. **Implementar `src/drift/artifacts.py`.**
   - `salvar_resultado(bloco, granularidade, escopo, metadata,
     resultados_df, base_path)`.
   - `carregar_resultado(caminho)`.
   - Testes: round-trip preserva todos os campos.

7. **Scripts em `scripts/drift/`** para cada bloco, parametrizando
   granularidade e escopo via CLI (`--granularidade mensal --escopo
   global`, etc.), iteráveis em loop simples para cobrir todos os
   cruzamentos.

8. **Testes de integração**: pipeline completo sobre subset pequeno do
   corpus (ex: 6 meses) verifica que todos os blocos produzem artefatos
   com schemas válidos.

9. **Notebook Colab adapter** em `notebooks/colab/drift_detection.ipynb`
   — mount do Drive, clone do repo, instalação via `requirements.txt`,
   execução dos scripts. Nenhuma lógica de drift dentro do notebook
   (regra §5 do CLAUDE.md).

10. **Verificação prévia da reprodutibilidade do código do Wanderley**
    (https://github.com/GDSMN/STIL2025_conceptdrift) — clonar, rodar,
    confirmar que conseguimos reproduzir pelo menos uma figura antes de
    nos comprometermos com a interpretação dos métodos.

### Revisão futura esperada

- Se *mean-pooling* mostrar resultado materialmente diferente em
  diagnóstico rápido, reabrir D.1 e consolidar o que for melhor.
- Se a penalidade BIC do CPD produzir muitos ou nenhum *change point*,
  ajustar (Pelt com penalidade fixa, ou número fixo de breakpoints).
- Se mensal não capturar drift que bi-semanal capture (ou vice-versa),
  reordenar primário/secundário no artigo.
- Se análise condicional `não-mercado` mostrar drift dominante e
  `mercado` mostrar estável, isso entra na narrativa como achado
  central e possivelmente reorienta a discussão da queda de F1(t).

## Referências

- Wanderley et al. STIL 2025 (PDF local
  `37849-769-30943-1-10-20251022.pdf`) — protocolo de referência.
- Reimers, N. & Gurevych, I. (2019). Sentence-BERT — para a discussão
  de pooling no contraponto. [CITAÇÃO PENDENTE: verificar entrada
  bibliográfica completa antes de incluir no artigo].
