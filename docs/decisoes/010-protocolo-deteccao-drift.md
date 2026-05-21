---
numero: 010
slug: protocolo-deteccao-drift
data: 2026-05-20
ciclo: pré-experimentação
status: aceita
depende_de: [003, 004, 008, 009]
---

# 010 — Protocolo de detecção e caracterização de drift no FolhaUOL

## Contexto

Drift no FolhaUOL **já foi parcialmente documentado** em ADRs anteriores,
mas exclusivamente no eixo de **rótulos** (P(y)):

- **ADR 003** estabeleceu cobertura temporal efetiva (33 meses, 2015-01 a
  2017-09) e reportou Mann-Kendall sobre série mensal de prevalência de
  `mercado` com τ = 0,49 e p < 1e-4 — evidência forte de tendência
  monotônica.
- **ADR 004 (§D.2)** estendeu o diagnóstico para todas as categorias:
  Mann-Kendall significativo em **24 de 48** classes; `mercado` sobe de
  10,6% (2015) para 13,1% (2017); `colunas` e `ilustrada` crescem;
  `esporte` colapsa em 2017.
- **ADR 008** elevou esse drift a **resultado do artigo**: o perfil
  F1(t) de `mercado` ao longo dos 5 folds *rolling-origin* é a figura
  central, e o contraste contra *stratified k-fold* sem *grouping* é a
  ablation que quantifica o otimismo de protocolos que ignoram tempo.

O que **ainda falta caracterizar**:

1. **Drift representacional** — a distribuição de embeddings P(x) das
   notícias muda ao longo dos 33 meses, ou apenas a proporção entre
   classes (P(y)) muda? Sem isso, não distinguimos *prior shift* puro de
   *covariate shift*.
2. **Natureza semântica do drift** — assumindo que P(x) muda, *o quê*
   muda? Tópicos novos entram? Vocabulário desloca? Eventos externos
   (eleições municipais 2016, impeachment 2016, crise política 2017)
   deixam assinatura mensurável?
3. **Momentos específicos** — drift é gradual ou tem pontos de mudança
   identificáveis? Mann-Kendall mede tendência, não localização de
   shifts.
4. **Drift intra-classe** — o que é "notícia de mercado" em 2015 é o
   mesmo que em 2017, ou a classe positiva muda de composição interna?

Sem esses quatro pontos, a seção de "caracterização do drift" do artigo
fica reduzida a "Mann-Kendall mostrou tendência" + "F1(t) cai" —
descritivo, não explicativo. Revisor BRACIS razoavelmente pedirá
profundidade maior, dado que **drift é a contribuição central do
artigo** (consequência herdada da ADR 008).

O artigo **Wanderley et al. STIL 2025** (`A Moving Target: Detecting
Concept Drift in Brazilian Portuguese Fake News`,
`37849-769-30943-1-10-20251022.pdf`) oferece um protocolo metodológico
recente, em PT-BR, com três blocos complementares:

- **B1 — Testes estatísticos de duas amostras** sobre embeddings:
  Kolmogorov-Smirnov (KS), Cramér-von Mises (CVM), Kernel Two-Sample
  Test (KTS), Least-Squares Density Difference (LSDD). Univariados
  agregados via método de Fisher; KTS e LSDD nativamente multivariados.
- **B2 — Análise semântica**: distância cosseno entre centróides de
  janelas, versão cumulativa contra média histórica, MMD² e WMD (esta
  última só com embeddings estáticos).
- **B3 — Change Point Detection (CPD)**: *binary segmentation* com
  custo kernel gaussiano sobre série de centróides diários reduzida a
  1ª PC via PCA.

Com **baseline aleatorizado** (partições embaralhadas) e **5
repetições** para isolar drift de variância amostral.

Esta ADR decide **quais desses blocos adotar** como protocolo de
caracterização de drift para o FolhaUOL. Parâmetros concretos (tamanho
de janela, escolha entre BERTimbau e word2vec, thresholds, número de
repetições) são **adiados para ADR seguinte** após confirmação dos
blocos.

## Alternativas consideradas

| Opção | Blocos | Análise por classe | Custo relativo | Profundidade | Comentário |
|---|---|---|---|---|---|
| 1 | manter só Mann-Kendall (ADR 003/004) | — (global) | mínimo | superficial | Insuficiente: caracteriza P(y), não P(x) nem momentos. |
| 2 | + B1 (estatístico) | global | médio | parcial | Responde "tem drift representacional?", não "o quê" nem "quando". |
| 3 | + B1 + B2 (estatístico + semântico) | global | médio-alto | melhor | Falta localização de shifts. |
| 4 | + B1 + B3 (estatístico + CPD) | global | médio-alto | melhor | Falta caracterização da natureza do drift. |
| 5 | + B1 + B2 + B3 (replica Wanderley) | global | alto | completo | Reproduz protocolo conhecido em PT-BR, mas ignora especificidade da tarefa de classificação temática. |
| **6** | **B1 + B2 + B3 + análise condicional por classe** | **`mercado` vs resto, separados** | **alto** | **completo + ajustado à tarefa** | **Recomendada.** |

Opção 1 está descartada implicitamente pelo problema. Opções 2-4 cobrem
subconjuntos próprios da 5/6 e ficam disponíveis como *fallback* se o
custo de 5/6 se mostrar inviável no Colab.

## Decisão

**Opção 6 — três blocos completos com análise condicional por classe.**

Confirmada pelo aluno em 2026-05-20.

### D.1 Blocos adotados

1. **B1 — Testes estatísticos sobre embeddings.** Adotar os quatro
   testes do Wanderley:
   - **KS** e **CVM** como detectores univariados de baixa sensibilidade
     (servem de controle: se *eles* disparam, drift é forte).
   - **KTS** e **LSDD** como detectores multivariados sensíveis (são os
     que mais provavelmente respondem afirmativamente).
   - Agregação de p-valores univariados via **método de Fisher** quando
     aplicável.
   - **Baseline aleatorizado** (partições embaralhadas sem reposição)
     com **5 repetições**, igual ao Wanderley, para isolar drift de
     variância amostral.

2. **B2 — Análise semântica.**
   - **Distância cosseno entre centróides** consecutivos e cumulativa
     contra média histórica.
   - **MMD²** sobre embeddings BERTimbau (a ADR 009 já compromete
     BERTimbau como família primária, então o embedding está disponível
     sem custo adicional).
   - **WMD** fica **opcional**: só vale se decidirmos treinar word2vec
     paralelamente. Sem isso, WMD é incompatível com BERTimbau
     contextual, como o próprio Wanderley nota.

3. **B3 — Change Point Detection.**
   - *Binary segmentation* com kernel gaussiano via biblioteca
     `ruptures`.
   - Sinal: centróides agregados em granularidade a definir (diária
     como no Wanderley, semanal, ou mensal — pendente ADR de
     parâmetros), projetados em 1ª componente PCA para reduzir a série
     univariada.
   - Objetivo: localizar shifts identificáveis (eleições out/2016,
     impeachment ago/2016, mudança de governo, eventos macroeconômicos)
     como ancoragem narrativa para a discussão do artigo.

### D.2 Análise condicional por classe

**Adição em relação ao Wanderley.** O Wanderley separa `true news` de
`fake news` porque a tarefa é binária e ambas as classes têm volume
comparável. No nosso caso, a tarefa primária é também binária projetada
(`mercado` vs resto, ADR 006), mas com desequilíbrio ~12,6% / 87,4%.

A análise condicional consiste em:

- Aplicar **B1, B2, B3** três vezes: sobre todo o corpus, só sobre
  `mercado`, e só sobre `não-mercado`.
- Permite separar:
  - **Drift global** (todos os documentos mudam) vs
  - **Drift intra-classe `mercado`** (a noção de "notícia de mercado"
    muda — relevante para a fronteira de decisão) vs
  - **Drift do "resto"** (mudança na composição do *bucket* residual —
    relevante porque o resto entra como negativo no treino binário).
- Resultado esperado: poder afirmar no artigo se a queda de F1(t) é
  explicada por mudança em `mercado`, em `não-mercado`, ou em ambos.

### D.3 Protocolo de comparação time-ordered vs randomized

Mantém o desenho do Wanderley:

- **Partições contíguas temporais** (granularidade a definir) vs
  **partições aleatorizadas sem reposição** (quebra ordem temporal).
- **5 repetições** independentes; cada bloco produz 5 valores; reporte
  é média ± desvio.
- A diferença estatística entre os dois cenários **é a evidência direta
  de drift** — variância de amostragem aleatória é o nulo.

### D.4 Itens deliberadamente fora desta ADR

- **Granularidade de janela** (bi-semanal como Wanderley, mensal
  alinhado com ADR 008, diária para CPD): **ADR seguinte**.
- **Embeddings concretos** (só BERTimbau? BERTimbau + word2vec? base
  ou large?): **ADR seguinte**, possivelmente integrada à ADR 009 se
  houver implicação cruzada.
- **Threshold de significância** (α = 0,05 como Wanderley? Bonferroni
  para múltiplas comparações?): **ADR seguinte**.
- **Implementação concreta** (estrutura de `src/drift/`, contrato de
  artefatos, testes): **ADR seguinte**, após decisão de parâmetros.
- **Drift condicional P(y|x)** — *concept drift* propriamente, no
  sentido de mudança da fronteira de decisão. Nenhum dos três blocos
  do Wanderley o mede diretamente; o sinal análogo no nosso protocolo
  é o **perfil F1(t) da ADR 008** já planejado. Esta ADR **não**
  adiciona detector de *concept drift* além do que a ADR 008 já
  contempla.

## Justificativa

1. **Os três blocos respondem perguntas complementares.** Adotar
   parcialmente abre lacunas óbvias que revisor BRACIS levantará:
   - Só B1 → "vocês detectaram drift mas não disseram o quê nem quando".
   - B1+B2 sem B3 → "qual o momento dos shifts?".
   - B1+B3 sem B2 → "qual a natureza semântica?".
   - Os três combinados produzem o quadro completo: detecção +
     caracterização + localização.

2. **Análise condicional por classe é necessária dada a tarefa.** O
   Wanderley trabalha com tarefa binária balanceada (fake vs true); o
   nosso desbalanceamento (12,6% / 87,4%) torna a média global
   dominada pelo "resto", potencialmente mascarando drift relevante em
   `mercado`. Separar é metodologicamente mais honesto e fornece
   subsídio direto para discussão da queda de F1(t).

3. **Custo é absorvível dentro do orçamento da ADR 009.**
   - Embeddings BERTimbau já são produzidos para a família 3 da ADR
     009 (D.1). Reusá-los para drift não exige *forward pass* extra
     além do necessário para inferência.
   - Testes estatísticos (KS, CVM, KTS, LSDD) rodam em CPU sobre
     vetores ≤1024-d em segundos por janela.
   - CPD via `ruptures` é leve (segmentação binária é O(n log n) no
     pior caso para n centróides diários ≤ 1.005).
   - Custo dominante seria treinar word2vec paralelamente — daí a
     opcionalidade do WMD em D.1 item 2.

4. **Compatibilidade com ADRs anteriores.**
   - **ADR 003**: drift de prevalência mensal já está documentado;
     esta ADR complementa, não substitui.
   - **ADR 006**: métrica primária (F1/PR-AUC binária de `mercado`)
     não muda. Drift detection é descritivo, não preditivo.
   - **ADR 008**: o perfil F1(t) continua sendo o resultado central;
     esta ADR fornece *contexto explicativo* para os movimentos do
     perfil.
   - **ADR 009**: embeddings BERTimbau e o pipeline já comprometidos
     são reusados sem alteração.

5. **Aderência a precedente metodológico em PT-BR.** O Wanderley
   STIL 2025 é a referência mais recente e específica para drift em
   texto PT-BR. Adotar o protocolo dele com adaptações justifica-se
   metodologicamente e facilita comparabilidade futura — em vez de
   inventar protocolo *ad hoc*, herdamos um já submetido a revisão por
   pares em conferência da área.

Contrapontos registrados (reconhecidos, não bloqueantes):

- **Não cobre concept drift P(y|x) diretamente.** O perfil F1(t) da
  ADR 008 é o substituto. Para BRACIS, a combinação "drift de X
  caracterizado + degradação de F1 ao longo do tempo" é narrativa
  defensável; pedir um detector formal de *concept drift* (tipo CDBD,
  DDM, EDDM) sairia do escopo. Se revisor pedir, ADR posterior pode
  adicionar.
- **CPD com binary segmentation tem hiperparâmetro de número de
  pontos.** O Wanderley usa o default da `ruptures` (penalidade BIC ou
  algo equivalente — verificar na implementação). Decisão de
  hiperparâmetro de CPD fica para a ADR seguinte; aqui adotamos o
  *método*, não a configuração.
- **Análise condicional triplica execuções de B1, B2, B3.** Aceito
  pelo ganho explicativo. Se Colab ficar apertado, *fallback* é
  reduzir a global + `mercado` (cortando "não-mercado", que é o menos
  informativo dos três).

## Consequências

### Habilita

- **Seção dedicada do artigo: "Caracterização do drift no FolhaUOL"**
  com três subseções (detecção estatística, evolução semântica,
  pontos de mudança) e análise paralela global vs `mercado`.
- **Figuras esperadas no artigo**:
  - Tabela com p-valores médios (com desvio) por teste em condição
    *time-ordered* vs *randomized*, análoga à Tabela 1 do Wanderley.
  - Série temporal de distância cosseno e MMD² entre janelas
    consecutivas, análoga à Figura 3 do Wanderley.
  - Marcação dos *change points* detectados sobre série de centróides,
    análoga à Figura 5 do Wanderley.
  - Versão *side-by-side* global vs `mercado` para cada figura.
- **Ancoragem narrativa**: pontos de mudança detectados pelo CPD podem
  ser confrontados com eventos externos conhecidos (eleições, crise
  política), reforçando a interpretabilidade.
- **Resposta direta à pergunta inicial do aluno sobre "como medir data
  drift"** (`docs/analise-previa/perguntas_iniciais/pergunta.txt`).

### Impede ou torna custoso

- **Triplica volume de execuções de drift detection** (global,
  `mercado`, `não-mercado`). Mitigação: testes são leves; o custo
  marginal é dominado pelo *forward pass* dos embeddings, que já é
  pago pela ADR 009.
- **Pesquisa de WMD** fica condicionada a ter word2vec no pipeline.
  Não comprometemos word2vec só por WMD: se a ADR seguinte decidir não
  treinar word2vec, WMD sai do escopo automaticamente, sem perda
  estrutural (MMD² cobre a função discriminativa).

### Dependências com outras ADRs

- **Herda** embeddings BERTimbau da ADR 009 (sem custo adicional).
- **Complementa** o diagnóstico de drift de prevalência da ADR 003/004
  com drift representacional e semântico.
- **Fornece contexto** para a interpretação da figura central do
  artigo (perfil F1(t) por fold, ADR 008).
- **Precede** uma ADR 011 sobre parâmetros e implementação de drift
  detection (janelas, embeddings, thresholds, contrato de artefatos,
  estrutura de `src/drift/`).

### Revisão futura esperada

- Se o protocolo de parâmetros (ADR 011) inviabilizar análise
  condicional por classe (volume insuficiente em `mercado` por
  janela), revisar para *fallback* "só global + `mercado` agregado".
- Se CPD não identificar nenhum *change point* significativo,
  reportar como achado negativo (drift no FolhaUOL é gradual, não
  abrupto) e ajustar narrativa.
- Se KTS/LSDD não rejeitarem o nulo nem em condição *time-ordered*,
  revisitar premissa de que há drift representacional — o drift
  documentado em ADR 003/004 pode ser de P(y) apenas, com P(x)
  estável. Esse resultado seria por si só uma contribuição.

### Pendências de implementação

Todas dependem da ADR 011 (parâmetros) antes de codificar. Em alto
nível:

- Módulo `src/drift/` com submódulos por bloco (`statistical.py`,
  `semantic.py`, `cpd.py`).
- Função de geração de janelas temporais com granularidade
  parametrizável e função paralela de partições aleatorizadas.
- Função de execução do protocolo de 5 repetições com agregação
  (média ± desvio) por teste e por condição.
- Contrato de artefatos de drift (formato de saída, naming, hashes) —
  detalhe na ADR 011.
- Testes unitários: determinismo das partições aleatorizadas sob
  *seed* fixa, sanidade das estatísticas em sinal sintético com drift
  conhecido (ex.: shift gaussiano em meio da série).

## Referência

- Wanderley, M. G., Ferraz, L. B. S., Almeida, T. A., Silva, R. M.
  (2025). *A Moving Target: Detecting Concept Drift in Brazilian
  Portuguese Fake News*. STIL 2025. PDF local:
  `37849-769-30943-1-10-20251022.pdf`. Código:
  https://github.com/GDSMN/STIL2025_conceptdrift (verificação de
  reprodutibilidade fica como TODO antes de implementar).
- Métodos individuais (citações secundárias herdadas do Wanderley,
  conferir antes de incluir no artigo):
  - KTS: Gretton et al. 2012.
  - LSDD: Bu et al. 2018; Sugiyama et al. 2013.
  - WMD: Kusner et al. 2015.
  - CPD com kernel: Garreau & Arlot 2017; Truong et al. 2020.
  - Benchmark de detectores em texto: Feldhans et al. 2021;
    Feldhans & Hammer 2025.
