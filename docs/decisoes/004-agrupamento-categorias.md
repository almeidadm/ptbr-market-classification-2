---
numero: 004
slug: agrupamento-categorias
data: 2026-04-24
ciclo: inicial
status: aceita
---

# 004 — Agrupamento e filtro de categorias

## Contexto

O corpus FolhaUOL (decisão 001) tem **48 categorias** distribuídas de forma fortemente
desigual sobre 167.053 notícias. O estudo deste projeto contempla **dois regimes**:

1. **Binário** — `mercado` vs. `não-mercado` (`eh_mercado`, `src/eda/carregamento.py:35`).
2. **Multiclasse** — com avaliação *one-vs-rest* do score individual de `mercado`
   após o treino multiclasse.

Ambos os regimes compartilham o corpus e o pré-processamento; diferem apenas no
rótulo-alvo. Essa é a primeira decisão sobre **como tratar as categorias
não-alvo** — tanto as grandes (que competem semanticamente com `mercado`) quanto
a cauda longa (onde cabem desde classes médias como `opiniao`/`tec` até
resíduos efêmeros com 1-3 amostras).

A decisão tem consequências metodológicas imediatas:

- Define o corpus utilizável (que afeta prevalência da classe positiva, e
  portanto métricas).
- Define o conjunto de classes do multiclasse (que afeta a fronteira de decisão
  e a calibração de `P(mercado | x)`).
- Define o que entra em análises de erro (onde os FPs e FNs podem ser
  rastreados até a categoria-fonte).
- Define o grau de comparabilidade com Garcia et al. (2024) como baseline
  externa.

## Dados que informam a decisão

### D.1 Forma da distribuição de categorias (§1)

| Posição | Categoria | N | % | Cum. |
|---|---|---:|---:|---:|
| 1 | poder | 22.022 | 13,18% | 13,2% |
| 2 | colunas | 21.622 | 12,94% | 26,1% |
| 3 | mercado | 20.970 | 12,55% | 38,7% |
| 4 | esporte | 19.730 | 11,81% | 50,5% |
| 5 | mundo | 17.130 | 10,25% | 60,7% |
| 6 | cotidiano | 16.967 | 10,16% | 70,9% |
| 7 | ilustrada | 16.345 | 9,78% | 80,7% |
| 8-10 | opiniao, paineldoleitor, saopaulo | 12.491 | 7,48% | 88,2% |
| 11-48 | cauda (38 classes) | 19.776 | 11,84% | 100,0% |

Descontinuidade abrupta entre a 7ª e a 8ª posição (fator ~3,6×). Dentro da
cauda, **25 categorias têm menos de 500 amostras**; **19 têm menos de 100**;
**9 têm menos de 3 meses de cobertura**; extremos: `musica=1`, `bichos=1`,
`2015=1`, `2016=1`, `contas-de-casa=2`.

### D.2 Deriva temporal por categoria (§3 e §3b)

**24 de 48 categorias** têm tendência Mann-Kendall significativa a p<0,01; 21
a p<0,001 (`artifacts/eda/tabelas/03b-deriva-por-categoria.csv`). Destaques:

- **Crescedores:** `colunas` (τ=0,81, 9,7%→16,1%), `sobretudo` (0%→2,2%),
  `opiniao` (2,3%→3,0%), `mercado` (10,6%→13,1%), `ilustrada` (8,7%→11,9%).
- **Quedados:** `folhinha` (1,0%→0%), `comida` (1,2%→0%), `paineldoleitor`
  (3,5%→1,0%), `tec` (2,2%→0,5%), **`esporte`** (13,9%→6,5%), `bbc`
  (1,3%→0%).

A prevalência aparente de `mercado` sobe em parte porque o **mix editorial
encolhe ao redor** — `colunas` e `ilustrada` crescem mais rápido, `esporte`
colapsa. Deriva composicional é a regra.

### D.3 Sobreposição semântica `mercado` × cauda (§6b)

Valores de Jaccard top-30 em `artifacts/eda/tabelas/06b-jaccard-mercado-vs-todas-categorias.csv`.
**Ressalva:** os valores não são diretamente comparáveis aos do §6 original
(IDF do TF-IDF muda com o conjunto de categorias avaliadas). O ranking
**interno** desta análise é válido.

Definição operacional adotada aqui: **"quase-mercado"** = categorias da cauda
com Jaccard ≥ 0,10 contra `mercado`. Volume total: **10.941 artigos** em 8
categorias.

| Categoria | N | Jaccard | Observação |
|---|---:|---:|---|
| opiniao | 4.525 | 0,111 | classe grande, τ positivo |
| tec | 2.260 | 0,132 | classe grande, 931 pares near-dup internos (41% da categoria) |
| tv | 2.142 | 0,154 | classe grande, Jaccard alto |
| bbc | 980 | 0,132 | quase descontinuada em 2016-17 |
| asmais | 548 | 0,111 | descontinuada |
| seminariosfolha | 379 | 0,154 | cresce em 2017 |
| banco-de-dados | 64 | 0,111 | nova em 2017 |
| cenarios-2017 | 43 | 0,132 | só 2016 |

Esses são os alvos principais de uma política de agrupamento: o que fazer com
eles determina a contaminação do bucket 'outros' ou da classe negativa.

### D.4 Duplicatas por categoria (§7)

Fração do corpus afetada por duplicatas (exatas ∪ near) concentra-se
assimetricamente:

- **Duplicatas exatas de `title` (4.015 linhas):** `ilustrada=1.254`,
  `colunas=675`, `tv=624`, `cotidiano=373` — as três primeiras concentram 64%.
- **Duplicatas exatas de `text` (597 linhas):** `colunas=170`, `tv=85`,
  `mercado=67`, `tec=61`, `ilustrissima=41`.
- **Near-duplicates intra-categoria:** `colunas↔colunas=1.127`,
  `tec↔tec=931`, `mercado↔mercado=178`, `tv↔tv=146`.
- **Pares cross-classe envolvendo mercado:** 39 (contrapartes dominantes: `tv=17`,
  `tec=6`, `poder=5`).

Lição: um filtro que remova `colunas + ilustrada + tv + tec` elimina a maior
parte da contaminação por duplicatas antes mesmo de invocar política de
deduplicação.

### D.5 Impacto quantitativo por opção (este ciclo)

Reaplicando a métrica de §7 ao corpus resultante de cada filtro, preservando
o conjunto de IDs afetadas (`carregar_articles` ordena por `(date, link)` —
os IDs em `07-near-duplicates-pares.csv` são pós-ordenação):

| # | Opção | N corpus | Prev. mercado | N classes | % dupl. | N quase-mercado |
|---|---|---:|---:|---:|---:|---:|
| 0 | Baseline | 167.053 | 12,55% | 48 | 2,058% | 10.941 |
| 1 | top-6 + mercado + outros (relabel; N=167k) | 167.053 | 12,55% | 8 | 2,058% | 10.941 em outros |
| 2 | top-9 + mercado + outros (relabel; N=167k) | 167.053 | 12,55% | 11 | 2,058% | 6.416 em outros |
| 3 | Garcia-style (5 classes) | 96.819 | 21,66% | 5 | 0,706% | 0 |
| 4 | Híbrido (−colunas, −ilustrada, −cauda<500) | 126.921 | 16,52% | 21 | 1,325% | 10.455 (classes próprias) |
| 5 | Threshold ≥500 | 164.888 | 12,72% | 23 | 2,077% | 10.455 (classes próprias) |
| 6 | Threshold ≥1.000 | 160.815 | 13,04% | 18 | 2,122% | 8.927 (classes próprias) |
| 7 | Threshold ≥1.000 + relabel top-6+mercado+outros | 160.815 | 13,04% | 8 | 2,122% | 8.927 em outros (34% do bucket) |

**Notas sobre Opção 7 (avaliada a pedido do usuário):**
- Corpus: 160.815 (96% do baseline). Descarta 6.238 artigos em 30 categorias
  com <1.000 amostras.
- 8 classes: `poder, colunas, mercado, esporte, mundo, cotidiano, ilustrada`
  (top-6 concorrentes + mercado) + `outros` (26.029, 16,19%).
- Categorias em `outros`: `opiniao, paineldoleitor, saopaulo, tec, tv,
  educacao, turismo, ilustrissima, ciencia, equilibrioesaude, sobretudo` —
  11 categorias agrupadas.
- **34% do bucket `outros`** (8.927/26.029) é "quase-mercado" (`opiniao, tec,
  tv`). Contaminação semântica substantiva.
- Taxa de duplicatas praticamente inalterada (2,12% vs. 2,06% baseline)
  porque mantém `colunas` e `ilustrada` como classes próprias — isso
  preserva o grosso do problema de duplicatas editoriais.

### D.6 Precedente metodológico: Santana, Oliveira & Nascimento (2022)

Santana et al. (2022), *Journal of Systemics, Cybernetics and Informatics*,
20(5), 33-59. DOI: [10.54808/JSCI.20.05.33](https://doi.org/10.54808/JSCI.20.05.33).
PDF em `docs/analise-previa/SA702PC22.pdf`. É o precedente em PT-BR mais
próximo metodologicamente do nosso problema de agrupamento de classes.

**Contexto do estudo deles:** classificação de notícias do jornal regional
"A Tribuna" (Vitória/ES, 2004-2007, 42.123 amostras em 19 categorias
desbalanceadas). ~77% do volume concentrado em 6 categorias. Menor
categoria: 30 amostras.

**Estratégia de agrupamento (§3 do paper, "Methodology"):**

1. Wordclouds por categoria para sinalizar coerência semântica.
2. **K-means sobre vetores TF-IDF** das notícias; número de clusters
   escolhido pelo **método do cotovelo com métrica Inertia** (Kodinariya &
   Makwana, 2013). Convergência: **6 clusters**.
3. Justificativa adicional: "almost 80% of instances were distributed in
   only six of nineteen existing categories" — análogo ao nosso top-7 com
   80,7% em §1.
4. **Heatmaps** da similaridade entre cada uma das 19 categorias originais
   e os 6 clusters majoritários, usados para decidir em qual cluster
   incorporar cada categoria minoritária.
5. **Resultado: 19 → 10 classes finais** — Economy, Politics, Style, Local
   News, Crime, Sports, International, Regional, Opinion, Tech.

**Resultados empíricos relevantes para nossa decisão** (Table 2, Fig.
19 do paper):

- Fine-tuned BERTimbau atingiu **F1 macro ≈ 0,86** nas 10 classes.
- Classe **"Economy"** (maior classe, ~15,6% de prevalência, análoga à
  nossa `mercado`): F1 = **0,85** — ordem de grandeza sugestiva do que é
  atingível para nossa `mercado` (prevalência 12,5%-21,7% conforme
  recorte).
- Best: "Sports" (F1 = 0,96); "Style" (0,93).
- **Pior classe em todos os modelos: "Regional"**. F1 saiu de 0,20
  (Word2Vec CBOW) → 0,54 (BERTimbau original) → 0,74 (fine-tuned), sempre
  20-22 p.p. abaixo das melhores.

**Leitura crítica — o que esse precedente empírico significa para 004:**

Mesmo um agrupamento **principiado** por clustering (K-means + TF-IDF +
elbow + heatmap, não arbitrário) produz pelo menos **uma classe sintética
estruturalmente fraca** — "Regional", que ainda assim foi o maior gargalo
de classificação com BERT fine-tuned. A fronteira de decisão da rede não
encontra estrutura semântica coerente quando os componentes fundidos são
heterogêneos demais. Isso é **evidência empírica direta** a favor da
preferência por filtrar (Família B) sobre agrupar (Família A) — e abre
uma opção intermediária (Opção 8, abaixo) de clustering-guided merge em
vez de bucket residual `outros`.

**Limitações do paper para transferência direta:**

- Corpus ~4× menor (42k vs 167k), período mais curto (4 anos vs. 33
  meses).
- **Não documentam quais categorias originais foram fundidas em quais
  classes finais** — o passo crítico do agrupamento é opaco, compromete
  reprodutibilidade.
- Sem análise de deriva temporal, sem análise de duplicatas, sem
  ablation "com agrupamento × sem agrupamento" nem "K=6 × K=8 × K=10".
- Sem avaliação one-vs-rest para classe de interesse específica — eles
  reportam per-classe mas não privilegiam nenhuma.
- "A Tribuna" é regional capixaba; generalização para FolhaUOL limitada.
- `[CITAÇÃO PENDENTE: verificar DOI e paginação da publicação antes de
  fechar o texto do artigo — checado em docs/analise-previa/ mas
  confirmar com resolução DOI ao citar formalmente]`.

## Alternativas consideradas

As sete opções acima, agrupadas por filosofia:

### Família A — preservar o corpus completo, tratar cauda pela rotulagem

**Opção 1** (top-6 + mercado + outros; 8 classes):
- *Prós:* Zero perda de volume. Estrutura mínima de classes para multiclasse.
- *Contras:* `outros` = 32.267 (19,3%), dos quais 34% são quase-mercado;
  herda a totalidade dos problemas de duplicatas e deriva composicional;
  `P(outros | x)` mal-definido por heterogeneidade interna.

**Opção 2** (top-9 + mercado + outros; 11 classes):
- *Prós:* Retira `opiniao, paineldoleitor, saopaulo` de `outros` (as três
  maiores e semanticamente ambíguas).
- *Contras:* Ainda mantém `tec, tv, bbc, asmais` em `outros` (6.416
  quase-mercado, 32% do bucket). Não resolve deriva nem duplicatas.

**Opção 7** (threshold ≥1.000 + relabel top-6 + mercado + outros; 8 classes):
- *Prós:* Estrutura de multiclasse enxuta (8 classes, como Opção 1) com
  saneamento mínimo da cauda (remove 6.238 artigos em categorias <1.000, que
  incluem os extremos não-treináveis como `musica=1`).
- *Contras:* `outros` = 26.029 (16,2%), dos quais 34% (`opiniao+tec+tv`) são
  quase-mercado; taxa de duplicatas praticamente inalterada (2,12%); mantém
  `colunas` e `ilustrada` como classes próprias (problema editorial intacto).
- **Leitura:** é uma Opção 1 mais enxuta e sem os resíduos extremos. A
  contaminação semântica e a impureza editorial persistem.

### Família B — filtrar a fonte, eliminar 'outros'

**Opção 3** (Garcia-style; 5 classes):
- *Prós:* Comparabilidade direta com Garcia et al. (2024) como baseline
  citável (`docs/analise-previa/Garcia et al. - 2024...pdf`, Table 2);
  corpus limpo (0,71% de duplicatas, zero quase-mercado); prevalência
  favorável (21,66%).
- *Contras:* Perde 42% do corpus (70k artigos descartados); classes
  balanceadas mas o problema vira de natureza diferente (5-classes balanceado
  ≠ classificação de 48 categorias); perde capacidade de caracterizar
  comportamento em corpus "real" do jornal com todo o mix editorial.

**Opção 4** (híbrido, −colunas −ilustrada −cauda<500; 21 classes):
- *Prós:* Remove explicitamente as duas categorias editoriais mais
  problemáticas (1.929 duplicatas de título combinadas) e a cauda
  não-treinável; `opiniao, paineldoleitor, saopaulo, tec, tv, etc.`
  permanecem como classes próprias (diagnóstico preservado no multiclasse);
  prevalência 16,5%.
- *Contras:* Perde 24% do corpus; 21 classes é muito para modelos clássicos
  (SVM, LogReg); não zera quase-mercado (10.455 artigos em classes
  próprias) — mas isso é intencional (permite rastrear FPs).

### Família C — filtrar por cardinalidade apenas, sem 'outros'

**Opção 5** (threshold ≥500; 23 classes):
- *Prós:* Mal reduz corpus (165k); estrutura multi-classe com 23 classes
  tratáveis.
- *Contras:* Não resolve duplicatas (2,08%) nem deriva nem quase-mercado;
  efeito essencialmente cosmético (remove só os extremos não-treináveis).

**Opção 6** (threshold ≥1.000; 18 classes):
- *Prós/Contras:* Similares à Opção 5, com 5 classes adicionais removidas.

### Família D — agrupamento orientado por clustering (K-means + TF-IDF)

**Opção 8** (clustering-guided merge; inspirada em Santana et al. 2022):

- *Proposta:* em vez de um único bucket `outros` (Família A) ou descarte
  da cauda (Família B), fundir a cauda em um número pequeno (3-5) de
  **classes semanticamente coerentes** usando o método de Santana et al.:
  K-means sobre TF-IDF das notícias da cauda, elbow para escolher K,
  heatmap categoria × cluster para vetting humano. Não-mercado no top-7
  + mercado ficariam como classes próprias.

- *Exemplos plausíveis de agrupamentos à luz dos dados já levantados*
  (especulativo — requer K-means real para confirmar):
    - `agencias-internacionais` = `bbc + dw + rfi + euronews` (~1.065
      artigos, semanticamente homogêneo: notícia internacional traduzida).
    - `estilo-vida` = `turismo + comida + folhinha + serafina + asmais +
      ilustrissima + sobretudo` (~5.500 artigos, heterogêneo mas todos
      próximos de "lifestyle/cultura").
    - `tecnologia-ciencia` = `tec + ciencia + cenarios-2017` (~3.650
      artigos).
    - `metro-local` = `saopaulo + o-melhor-de-sao-paulo` (~4.140
      artigos, ambos São Paulo local).
    - `educacao-saude-social` = `educacao + equilibrioesaude +
      empreendedorsocial + ambiente` (~4.760 artigos).
  Total resultante: ~13-14 classes (top-7 + mercado + 5 sintéticas +
  descartar resíduos < ~50 amostras). Todas trateáveis por modelos
  clássicos e transformers.

- *Prós:*
    - Classes sintéticas mais **coerentes semanticamente** que um
      `outros` amálgama — reduz o risco de P(outros | x) achatado por
      heterogeneidade (problema descrito em V.2 da análise EDA).
    - **Diagnóstico de FPs preservado em nível de grupo**: se um FP de
      mercado vier de `estilo-vida`, sabe-se que não veio de
      `agencias-internacionais`. Granularidade inferior a "classe
      própria" mas superior a "outros".
    - Precedente metodológico publicado (Santana et al. 2022) — citável
      e defensável em revisão BRACIS.
    - Reduz cardinalidade de 48 → 13-14 — tratável para SVM, LogReg,
      BERT.
    - Preserva volume relativamente alto do corpus (descartar só resíduo
      < 50 amostras ≈ 30-50 artigos).

- *Contras:*
    - **Exige cycle próprio** de experimento (clustering + vetting
      humano dos grupos com a orientadora), não está quantificada nesta
      tabela.
    - Os grupos propostos acima são especulação baseada em conhecimento
      de mundo; o K-means real pode sugerir fronteiras diferentes e
      menos intuitivas (risco de obter uma "classe regional
      artificial").
    - **Evidência empírica do próprio Santana et al. mostra que classes
      obtidas por clustering ainda sofrem** — "Regional" foi a pior
      classe em todos os modelos deles, mesmo com metodologia
      principiada.
    - Não resolve duplicatas editoriais: se `colunas` e `ilustrada`
      ficam como classes próprias, o gargalo de 2,06% de duplicatas
      persiste (mesmo problema que Opções 1, 2, 5, 6, 7).
    - Exige justificação metodológica mais elaborada no artigo (cluster
      choice, K choice, vetting humano, defender por que esses
      agrupamentos).

- *Quantificação pendente:* a linha 8 da tabela D.5 ficaria em branco —
  o impacto quantitativo depende de quais categorias o K-means
  efetivamente agrupa. Se esta opção for escolhida como primária,
  abrir cycle próprio (decisão 004b ou 005a).

## Decisão metodológica secundária

Ortogonal à escolha entre 1-7 há uma segunda decisão de escopo:

- **(a) Corpus único para ambos regimes** (binário e multiclasse).
- **(b) Corpora distintos** — ex.: binário sobre corpus amplo (captura
  desbalanceamento realista), multiclasse sobre corpus filtrado (captura
  fronteira semântica com classes bem-definidas).

A Opção **(b)** requer justificativa adicional no artigo (por que dois
corpora? o que cada regime mede?), mas **permite** que o binário responda
"quão bem classificamos mercado em um cenário realista de imprensa" e o
multiclasse responda "quão separável é mercado das classes concorrentes
quando controlamos ruído editorial".

## Decisão

**Recorte primário: Opção 7** (threshold ≥1.000 + relabel top-6 concorrentes +
mercado + outros; 8 classes; N=160.815).

- Classes: `poder, colunas, mercado, esporte, mundo, cotidiano, ilustrada,
  outros`.
- Prevalência de `mercado`: 13,04%.
- Bucket `outros`: 26.029 artigos (16,19%), agrupando 11 categorias
  (`opiniao, paineldoleitor, saopaulo, tec, tv, educacao, turismo,
  ilustrissima, ciencia, equilibrioesaude, sobretudo`); 34% do bucket
  (`opiniao+tec+tv`) é quase-mercado.
- Decisão metodológica secundária: **(a) corpus único** aplicado a ambos
  regimes (binário de ablation e multiclasse primário conforme ADR 006).

**Ablations obrigatórias, reportadas em tabelas suplementares:**

- **Opção 4** (híbrido, −colunas −ilustrada −cauda<500; 21 classes;
  N=126.921) — quantifica o custo empírico do bucket `outros` contra
  preservação categorial. Gap pequeno em F1/PR-AUC de `mercado` valida
  Opção 7; gap grande vira evidência quantitativa de "quanto custa o
  catch-all" — contribuição em si.
- **Opção 3** (Garcia-style, 5 classes; N=96.819) — comparabilidade
  externa direta com Garcia et al. (2024).

**Preservação de metadados (obrigatória em todos os datasets):**

Dois rótulos por documento, carregados desde o pré-processamento e
intocados pelo resto da pipeline:

- `y_colapsado` — o rótulo visto pelo modelo (8, 21 ou 5 classes conforme
  recorte da vez).
- `y_original` — o rótulo original das 48 categorias FolhaUOL; **metadado,
  nunca consumido pelo modelo em nenhum momento (treino, validação, teste)**.

Essa dupla rotulagem preserva a rastreabilidade semântica dos FPs e FNs
de `mercado` mesmo sob colapso do catch-all. É pré-requisito das análises
de contaminação descritas no reenquadramento abaixo.

**Reenquadramento do eixo metodológico do artigo:**

O artigo deixa de ser enquadrado como "construção de um classificador de
`mercado` no FolhaUOL" — recorte derivativo a Garcia et al. (2024) — e
passa a ser:

> **Classificação de uma classe de interesse em presença de um bucket
> residual heterogêneo: diagnóstico, mitigação e custo.**

Consequências para a estrutura do artigo:

1. **Introdução** motiva o problema de *catch-all class* em corpora
   jornalísticos reais: seções residuais, experimentais e transitórias
   são regra, não exceção, em redações com taxonomia vasta.
2. **Metodologia** documenta o protocolo de preservação de `y_original`
   e a pipeline de diagnóstico post-hoc que ele habilita.
3. **Resultados** reportam métrica primária de `mercado` (ADR 006) sobre
   Opção 7, ablations 4 e 3 em tabelas suplementares, e três análises
   específicas de contaminação do `outros`:
   - **A1 — Composição de FPs de `mercado` por `y_original`.** Resposta
     direta à pergunta da orientadora sobre análise semântica de FPs.
   - **A2 — Perfil de `P(mercado|x)` por sub-categoria de `outros`.**
     Identifica quantitativamente quais categorias originais "vazam"
     para `mercado` no espaço de decisão. Expectativa ancorada em D.3:
     `opiniao+tec+tv` dominam.
   - **A3 — Heterogeneidade interna de `outros` via variância de
     `P(outros|x)` condicionada em `y_original`.** Evidência quantitativa
     de que o bucket viola as condições de uma classe bem-definida.
4. **Discussão** articula o trade-off evidenciado: simplicidade de
   reporte (8 classes, Opção 7) × custo em F1/PR-AUC (via ablation
   Opção 4) × comparabilidade externa (via ablation Opção 3).

**Extensão opcional (se houver folga de tempo):** treinar um classificador
auxiliar leve (SVM ou LogReg) apenas sobre o subset de `outros` para
predizer as 11 categorias originais que o compõem. Fornece "diagnóstico
de segundo estágio" sobre a sub-estrutura de `outros` sem re-treinar o
modelo principal. Entra como apêndice, não como resultado central.

## Justificativa

1. **Reconciliação com a preferência por cardinalidade baixa.** 21
   classes (Opção 4) está na fronteira superior do tratável para modelos
   clássicos; 5 classes (Opção 3) é confortável mas perde 42% do corpus.
   Opção 7 com 8 classes fica no meio-termo — cardinalidade tratável em
   toda a família de modelos com perda de corpus mínima (3,7%).
2. **Opção 7 é estritamente superior a Opção 1.** Mesmas 8 classes, mesmo
   enquadramento narrativo, mas remove 6.238 artigos em categorias
   não-treináveis (`musica=1`, `bichos=1`, `2015=1`, `2016=1`,
   `contas-de-casa=2` etc.) que seriam puro ruído dentro do `outros`.
   Custo: 3,7% de volume; ganho: menos *garbage* no catch-all, threshold
   principiado e citável.
3. **O reenquadramento converte fraqueza em objeto de estudo.** Os cinco
   eixos de fraqueza da Opção 7 frente à Opção 4 (contaminação
   quase-mercado, duplicatas editoriais, deriva, calibração de
   `P(outros|x)`, evidência externa Santana 2022) deixam de ser objeções
   quando o artigo **é sobre diagnosticar exatamente esse comportamento**.
   Santana et al. (2022) fornece o precedente empírico da degradação de
   classes sintéticas mas **não formaliza o protocolo de diagnóstico** —
   essa é a lacuna endereçada.
4. **Ablations 4 e 3 antecipam as duas objeções principais em revisão.**
   - "Por que não preservaram a granularidade?" → Opção 4 quantifica o
     custo.
   - "Por que não usaram o corpus do comparativo?" → Opção 3 dá a linha
     Garcia na tabela.
5. **`y_original` tem custo marginal zero e retorno alto.** Uma coluna
   extra no DataFrame, zero impacto em treino, recuperação completa da
   rastreabilidade de FPs — exatamente o pedido da orientadora em
   `recomendacoes_orientadora.txt`.

Contrapontos registrados para consciência (reconhecidos e aceitos):

- **Duplicatas editoriais em `colunas` e `ilustrada` persistem** (2,12%
  do corpus). Política de dedup em ciclo futuro precisa ser específica
  para essas duas classes.
- **Análises A1-A3 precisam ser executadas com rigor.** Se ficarem
  superficiais (uma tabela de composição e acabou), o artigo volta a ser
  derivativo. O diferencial depende de execução.
- **Opção 8 (clustering-guided merge) continua sendo via metodologicamente
  superior em tese.** Fica registrada como ciclo candidato; não entra no
  escopo atual por exigir ciclo de experimento adicional.

## Recomendação do agente (histórico)

> **Nota (2026-04-24):** esta seção preserva a recomendação original do
> ciclo de rascunho, em que o agente recomendou Opção 4 como primária com
> Opção 3 como ablation. A decisão final acima (§Decisão) divergiu após o
> usuário explicitar desconforto com cardinalidade >10 e após
> reformulação do eixo metodológico para *"catch-all como objeto de
> estudo"*. A recomendação original continua válida como leitura técnica
> do trade-off (Opção 4 é mais forte em cinco eixos empíricos documentados
> em D.1-D.6), mas o enquadramento do artigo tornou a Opção 7 a escolha
> preferível. Opção 4 permanece no desenho como ablation, não como
> recorte descartado.

**Família B (filtro a montante) em vez de Família A (bucket 'outros').**
Especificamente, **Opção 4 como escolha primária**, com **Opção 3 como
ablation secundária** para comparabilidade com Garcia et al. Decisão
metodológica secundária: **(a) corpus único** (Opção 4 aplicada a ambos
regimes).

Justificativa da recomendação:

1. **Os três problemas estruturais do corpus (deriva composicional,
   duplicatas editoriais, quase-mercado) se resolvem melhor por filtro que
   por agrupamento.** Opção 7 — que você pediu para avaliar — ilustra o
   ponto: mesmo agrupando para 8 classes como na Opção 1 com um filtro de
   limiar moderado, a taxa de duplicatas e a contaminação semântica de
   `outros` permanecem próximas ao baseline.
2. **O bucket `outros`, onde existir, viola as condições de uma classe
   bem-definida:** heterogeneidade semântica alta, não-estacionariedade
   temporal alta, rastreabilidade diagnóstica baixa. Isso penaliza a
   avaliação one-vs-rest de `mercado` que é parte central do estudo.
   **Evidência empírica externa** (D.6): Santana et al. (2022) aplicaram
   clustering principiado (K-means+TF-IDF+elbow) para fundir cauda, e
   ainda assim a classe sintética mais heterogênea ("Regional") ficou
   consistentemente pior em todos os modelos, inclusive BERT
   fine-tuned (F1 0,74 vs. ≥ 0,78 nas demais). Se um agrupamento
   principiado produz classes fracas, um bucket residual `outros`
   (Família A) é ainda pior.
3. **Opção 4 preserva diagnóstico** (21 classes → cada FP rastreável),
   **reduz duplicatas sem eliminar o corpus** (1,33% vs. 2,06%), e
   **mantém os "quase-mercado" visíveis** (como classes próprias, não
   diluídos em 'outros'). Isso alinha com a recomendação da orientadora
   de análise semântica de FPs.
4. **Opção 3 como ablation** dá comparabilidade direta com Garcia et al.
   sem exigir que seja o recorte primário. No artigo, uma linha de tabela
   a mais; metodologicamente, defende contra a objeção BRACIS "por que
   não usaram o corpus do comparativo".

Argumentos contra a recomendação (honestidade):

- 21 classes é muito para modelos clássicos (SVM, LogReg). Se o pipeline
  experimental priorizar esses modelos, Opção 3 ou Opção 7 seriam mais
  práticas.
- Perder 24% do corpus é material; se o aluno quiser defender "estudamos o
  corpus inteiro do FolhaSP 2015-2017", a Opção 4 (ou 3) fere essa
  narrativa. Opção 7 preserva 96%.
- A escolha do limiar (<500) é arbitrária — poderia ser <200, <300,
  <1.000; cada variante gera resultados diferentes. Alternativa: declarar
  o limiar como hiperparâmetro e reportar sensibilidade.
- **Crítica Santana-style (Opção 8 como alternativa):** existe uma via
  intermediária (clustering-guided merge) que não foi quantificada por
  exigir experimento adicional. Se o objetivo for **máxima preservação
  de volume acoplada a estrutura semântica mais coerente** que `outros`
  amálgama, o caminho é Opção 8, não Opção 4. Minha resposta: (i) os
  dados empíricos do próprio Santana et al. (2022) mostram que mesmo
  clustering principiado produz uma classe fraca ("Regional"), o que
  ameniza a vantagem esperada; (ii) Opção 4 dá rastreabilidade máxima
  de FPs (categoria original preservada); (iii) Opção 8 não resolve
  duplicatas editoriais se mantém `colunas`/`ilustrada` como classes
  próprias. Portanto: **Opção 4 continua como primária, Opção 8 entra
  como cycle futuro candidato se 4 for rejeitada e o projeto preferir
  preservação de volume**.

## Consequências

### Habilita

- **Corpus operacional fixo** — 160.815 artigos, 8 classes, prevalência
  de `mercado` em 13,04%. Permite fechamento dos ciclos subsequentes
  (dedup, split, experimento).
- **Resolve a dependência da ADR 006** (setup multiclasse + métrica binária
  projetada): o espaço de argmax passa a ser concreto (8 classes Opção 7
  no recorte primário; 21 na ablation Opção 4; 5 na ablation Opção 3).
- **Análise semântica de FPs de `mercado`** via `y_original`, pedida pela
  orientadora — cada FP rastreável às 48 categorias originais, mesmo no
  regime colapsado.
- **Três análises de contaminação do catch-all** (A1, A2, A3 na §Decisão)
  como núcleo metodológico do artigo, não mero diagnóstico acessório.
- **Ablations 4 e 3** como tabelas suplementares, não como recortes
  primários concorrentes. Cada ablation responde a uma objeção
  antecipável de revisor.
- **Reformulação do título, abstract e introdução** em torno do eixo
  *"classificação com catch-all heterogêneo"*.

### Impede ou restringe

- Afirmar *"usamos todo o corpus FolhaSP 2015-2017"* — o texto correto
  passa a ser *"aplicamos um threshold de ≥1.000 amostras por categoria
  para remover resíduos não-treináveis"*. Afeta 3,7% do volume.
- Discriminação fina entre sub-categorias internas a `outros` pelo modelo
  principal — colapsada no gradiente. Recuperável parcialmente via
  modelo auxiliar de segundo estágio (extensão opcional em §Decisão),
  não via post-hoc direto do modelo principal.
- Cálculo de `P(tec|x)`, `P(tv|x)`, etc. pelo modelo principal — não
  existem. Mesma observação sobre modelo auxiliar.
- **Duplicatas editoriais em `colunas` e `ilustrada` persistem** (2,12%
  do corpus). Ciclo futuro de dedup precisa tratar especificamente essas
  duas classes — são ~65% do volume de duplicatas exatas de título.

### Dependências com outras ADRs

- **Resolvida:** ADR 006 (métrica primária e setup de treino) — o conjunto
  concreto de classes fica fixo e a ADR 006 entra em regime plenamente
  operacional.
- **Não altera:** ADR 005 (validação externa WikiNews, em `proposta`,
  postergada por decisão do aluno). Mapeamento WikiNews → binário segue
  válido sem ajuste.
- **Condiciona ciclo futuro de deduplicação:** política precisa considerar
  que `colunas` e `ilustrada` concentram o grosso das duplicatas e
  permanecem como classes próprias.
- **Condiciona ciclo futuro de split (temporal vs. estratificado):**
  deriva composicional persiste — `colunas` e `ilustrada` crescem, `esporte`
  colapsa em 2017. Medição de deriva no corpus Opção 7 fica como
  pré-requisito daquele ciclo.

### Custo experimental esperado

Cada família de modelo (SVM, BERTimbau, ...) requer, no mínimo, quatro
execuções para cobrir o desenho:

1. **Primária** — multiclasse Opção 7 (8 classes, ADR 006).
2. **Ablation ADR 006** — binário puro sobre rótulo `eh_mercado` do
   corpus Opção 7.
3. **Ablation Opção 4** — multiclasse, 21 classes, corpus 126.921.
4. **Ablation Opção 3** — multiclasse, 5 classes, corpus 96.819
   (Garcia-style).

Custo tratável em Colab para modelos clássicos; para transformers
(BERTimbau fine-tuning), pode forçar escolha entre número de famílias
testadas vs. número de ablations por família. Decisão a ser tomada no
ciclo de setup de experimento.

### Revisão futura esperada

- Se a ablation Opção 4 mostrar gap grande em F1/PR-AUC de `mercado`
  contra Opção 7 (ex.: >3 p.p.), **revisitar esta ADR**: a evidência
  empírica pode exigir promover Opção 4 a recorte primário.
- Se a Opção 8 (clustering-guided merge, Santana-style) for investigada
  em ciclo futuro e seus clusters resultarem ≤10 classes, pode
  substituir Opção 7 como primária — registrada como ciclo candidato,
  não descartada.
- **Sensibilidade ao threshold ≥1.000.** Fica como análise opcional no
  artigo; se revisor BRACIS questionar o corte, rodar com ≥500 e ≥2.000
  e mostrar que os resultados principais não dependem criticamente do
  valor.

### Pendências documentais

- **Literatura sobre "catch-all class" / "other bucket" / "open-set
  classification"** — `[CITAÇÃO PENDENTE: buscar literatura de 2019-2024
  sobre degradação de classes residuais sintéticas em NLP supervisionada;
  verificar se há formalização teórica da calibração de P(outros|x) sob
  heterogeneidade; candidatos iniciais: Geng, Huang & Chen 2020 "Recent
  advances in open set recognition"; Scheirer et al. 2013 "Toward open
  set recognition".]` — essa base é necessária para sustentar o
  reenquadramento metodológico em revisão.

## Revisão em 2026-04-24

Ajustes decorrentes da consolidação da ADR 007 (política de deduplicação).

1. **Predição sobre "ciclo futuro de dedup" não se realizou como
   esperado.** Em §Consequências/"Impede ou restringe" e
   §Consequências/"Dependências com outras ADRs", esta ADR previa que o
   ciclo futuro de deduplicação precisaria tratar especificamente
   `colunas` e `ilustrada`, que concentram o grosso das duplicatas de
   `title` exatas. A ADR 007 abriu o ciclo em 2026-04-24 e, por decisão
   explícita do aluno, **não** tratou essas classes: removeu apenas
   duplicatas exatas por `text` (597 linhas, 0,36% do raw) e preservou
   duplicatas exatas de `title` (4.015 linhas) e *near-duplicates*.
   Consequência: o problema editorial de `colunas`/`ilustrada` **persiste
   por desenho**, não por omissão. Eventual revisão desse quadro fica
   como ciclo candidato futuro, dependente de sinais empíricos durante
   experimentação (p.ex. modelo memorizando assinatura de colunistas).

2. **Volume do corpus Opção 7 ajustado marginalmente.** A §D.5 e a
   §Decisão reportam N=160.815 sob a contagem pré-dedup. Após a
   aplicação da ADR 007, o valor pós-dedup é aproximadamente
   **N=160.218** (variação de ~0,37%). O ajuste é pequeno o suficiente
   para não alterar nenhuma das análises qualitativas desta ADR
   (prevalência de `mercado` em ~13,04%, 34% do `outros` como
   quase-mercado, etc. permanecem válidos dentro do erro de
   arredondamento).

3. **Nova restrição herdada para o ciclo de splitting.** A ADR 007
   preserva *near-duplicates* (pares intra-classe em
   `colunas=1.127, tec=931, mercado=178, tv=146`). Isso impõe que o
   ciclo futuro de splitting adote estratégia *group-aware* — pares
   irmãos precisam permanecer no mesmo lado do split para evitar
   leakage. Essa restrição não existia quando esta ADR foi redigida e é
   registrada aqui para referência cruzada.

4. **Composição da Opção 3 formalizada.** A §D.5 registrava a Opção 3
   (Garcia-style) como N=96.819 em 5 classes mas não enumerava os
   rótulos. O aluno confirmou no ciclo de implementação (2026-04-24) a
   composição `{poder, mercado, esporte, mundo, cotidiano}` — os cinco
   maiores por volume no corpus FolhaUOL. `src/config.py` carrega
   `CLASSES_OPCAO_3` com essa tupla.

5. **Enumeração da Opção 4 materializada.** A §D.5 deixava em aberto
   quais categorias específicas compõem o recorte N=126.921 de 21
   classes; apenas o critério (`≥500` amostras, exceto `colunas` e
   `ilustrada`) estava fixado. `scripts/preprocessar.py` executa essa
   enumeração automaticamente sobre o corpus pós-dedup. As 21 classes
   obtidas: `asmais, bbc, ciencia, comida, cotidiano, educacao,
   empreendedorsocial, equilibrioesaude, esporte, folhinha,
   ilustrissima, mercado, mundo, opiniao, paineldoleitor, poder,
   saopaulo, sobretudo, tec, turismo, tv`. Os artefatos
   `data/processado/enumeracao_opcao4.json` e
   `enumeracao_opcao4_contagens.csv` documentam a saída para auditoria.

6. **Correção de implementação na Opção 7 (2026-04-24, segunda
   revisão).** A primeira versão de `src/preprocessamento/recortes.py:aplicar_opcao_7`
   omitiu o passo de `threshold ≥1.000` descrito na §D.5 item 7 e
   executava apenas o *relabel* — o que equivale à Opção 1, não à
   Opção 7. O bug foi identificado ao reconciliar os números
   pós-pipeline com esta ADR; `aplicar_opcao_7` passou a remover
   primeiro as categorias abaixo do threshold antes de colapsar o
   restante em `outros` (ordem correta segundo §D.5). Nenhum
   experimento real tinha rodado antes da correção; o único artefato
   produzido (smoke validation de LogReg) permanece válido como
   verificação de pipeline mas não como resultado a reportar.

7. **Números definitivos pós-pipeline.** Tabela de reconciliação entre
   as estimativas declaradas na §D.5 e os valores obtidos pela execução
   de `scripts/preprocessar.py` sobre `articles.csv`
   (SHA-256 registrado em `data/processado/hashes.json`):

   | Métrica | §D.5 (estimado) | Pós-pipeline (obtido) | Delta |
   |---|---:|---:|---:|
   | Corpus bruto pós-filtro `date`/`text` nulos | — | 166.288 | — |
   | Corpus pós-dedup (ADR 007) | ~166.456 | 165.901 | −555 (−0,33%) |
   | Opção 7 (pós-threshold + relabel) | 160.815 | 159.681 | −1.134 (−0,71%) |
   | Opção 4 (≥500 excluindo colunas/ilustrada) | 126.921 | 126.647 | −274 (−0,22%) |
   | Opção 3 (Garcia-style) | 96.819 | 96.742 | −77 (−0,08%) |

   Todas as diferenças são < 1% e vêm da composição de dois efeitos
   não previstos explicitamente nas estimativas originais: (i) filtro
   de linhas com `date` ou `text` nulos em `src/preprocessamento/carregamento.py:preparar_corpus`
   (pré-requisito para o protocolo temporal da ADR 008); (ii)
   contabilização diferente de linhas duplicadas entre "removidas"
   (387) e "envolvidas em grupos" (597, vide item 2 desta revisão e R.6
   da ADR 009). Nenhum delta altera as conclusões qualitativas desta
   ADR (prevalência de `mercado` ~13%, 34% do `outros` como
   quase-mercado, etc.).
