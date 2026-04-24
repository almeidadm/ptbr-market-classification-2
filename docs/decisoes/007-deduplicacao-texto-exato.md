---
numero: 007
slug: deduplicacao-texto-exato
data: 2026-04-24
ciclo: pré-experimentação
status: aceita
---

# 007 — Remoção de duplicatas exatas por `text`, preservação de duplicatas exatas de `title` e de *near-duplicates*

## Contexto

A EDA documentada na ADR 002 e resumida em §D.4 da ADR 004 caracterizou três
tipos de duplicação no corpus FolhaUOL:

1. **Duplicatas exatas por `text`** — 597 linhas. Concentração: `colunas=170`,
   `tv=85`, `mercado=67`, `tec=61`, `ilustrissima=41`. Representam 0,36% do
   corpus raw (167.053 artigos).
2. **Duplicatas exatas por `title`** — 4.015 linhas. Concentração: `ilustrada=1.254`,
   `colunas=675`, `tv=624`, `cotidiano=373` (três primeiras = 64%).
   Representam 2,40% do corpus raw.
3. ***Near-duplicates* (MinHash/LSH, Jaccard aproximado)** — principais focos
   intra-classe: `colunas↔colunas=1.127`, `tec↔tec=931`, `mercado↔mercado=178`,
   `tv↔tv=146`. Pares cross-classe envolvendo `mercado`: 39 (contrapartes
   dominantes em `tv=17`, `tec=6`, `poder=5`).

Esta ADR consolida a política de **deduplicação do corpus raw**, antes da
aplicação do recorte Opção 7 (ADR 004). Trata explicitamente dos três tipos
acima.

A política se aplica a *todos* os recortes — primário Opção 7 e ablations
Opções 4 e 3 — porque opera sobre o corpus raw antes de qualquer relabel ou
filtro por cardinalidade.

## Alternativas consideradas

| Opção | Ação sobre `text` | Ação sobre `title` | Ação sobre near-dup | Benefício | Custo / risco |
|---|---|---|---|---|---|
| **A. Nenhuma dedup** | preserva | preserva | preserva | volume máximo; decisão reversível. | artefatos editoriais inflam prevalência; leakage entre splits; análise A1 (FPs por `y_original`) pode computar o mesmo texto em categorias diferentes como dois FPs distintos. |
| **B. Apenas `text` exato** *(esta)* | remove | preserva | preserva | remove os artefatos inequívocos (zero sinal adicional); perda marginal (~0,36%); conteúdo serial com `title` idêntico é preservado. | não trata near-dups (risco de leakage em split aleatório); não trata duplicatas de `title` semanticamente suspeitas. |
| **C. `text` + `title` exatos** | remove | remove | preserva | remove todos os dups determinísticos. | 2,40% do corpus descartado — inclui conteúdo serial legítimo (colunas diárias, panoramas, placares) que compartilha `title` mas difere em `text`. |
| **D. `text` + `title` + near-dups** | remove | remove | remove | máxima pureza; mitiga leakage automaticamente. | agressivo; exige política de threshold de similaridade (Jaccard, cosine) que é em si um ciclo de decisão; remove ~5-6k linhas. |
| **E. Soft-dedup via flag apenas** | marca | marca | marca | totalmente reversível; permite experimentos sensíveis a dedup. | nenhuma dedup física aplicada; métricas ainda contaminadas a menos que todo pipeline downstream filtre explicitamente. |

## Decisão

**Opção B** — remoção física de duplicatas exatas por `text`, com:

1. **Critério de desempate.** Manter a cópia com `date` mais antiga.
   Empates em `date` resolvidos pelo `link` lexicograficamente menor
   (desempate determinístico, independente de seed).
2. **Ordem de operações.** Dedup aplicada sobre o **corpus raw de 167.053
   artigos** antes do threshold Opção 7 e antes do relabel. Justificativa:
   (i) bucket `outros` fica definido sobre corpus saneado; (ii) artigos que
   seriam duplicados em categorias cuja contagem cai abaixo do threshold
   deixam de existir como tais antes da contagem, evitando que uma
   categoria com 6 linhas (5 dups + 1 canônica) seja classificada como
   "cauda não-treinável" incorretamente.
3. **Coluna de auditoria `eh_duplicata_text`.** Preservada no dataset
   final (pós-dedup). Semântica: `True` se a linha é a sobrevivente
   canônica de um grupo de duplicatas exatas (i.e., havia ≥ 1 outra cópia
   no corpus raw com mesmo `text`, removida pelo desempate); `False` se a
   linha era única no raw. Permite auditoria, amostragem de casos e
   análise de sensibilidade sem exigir re-importação do raw.
4. **Duplicatas exatas de `title` preservadas integralmente.** Conteúdo
   serial (`ilustrada`, `colunas`, `tv`, `cotidiano` concentram o volume)
   compartilha cabeçalho mas tem corpo distinto na imensa maioria dos
   casos; é informação legítima, não artefato.
5. ***Near-duplicates* preservados integralmente.** Não há remoção baseada
   em MinHash/Jaccard neste ciclo. Registrado como restrição explícita do
   escopo desta ADR, não como omissão.

## Justificativa

- **`text` exato é artefato.** A coincidência de `text` byte a byte entre
  artigos distintos em corpus jornalístico indica re-publicação editorial,
  ingestão dupla, ou erro de deduplicação na fonte. Nenhuma dessas
  situações carrega sinal adicional para o classificador. Remoção é a
  ação correta, sem controvérsia.
- **`title` idêntico com `text` distinto é conteúdo serial.** Colunas
  fixas ("Painel", "Mônica Bergamo", etc.), panoramas, reportagens
  sequenciais compartilham `title` recorrente com corpo diferente a cada
  edição. Remover inflaria artificialmente a perda e descartaria exemplos
  bem-rotulados.
- **Volume removido é marginal.** 597 linhas em 167.053 (0,36%). O impacto
  em prevalência de `mercado` é desprezível (67 remoções em 20.970, ~0,32
  p.p. sobre a classe).
- **Critério de desempate "mais antiga" é convencional em limpeza de
  corpora jornalísticos** e evita injetar viés da re-publicação (a cópia
  mais antiga é a mais próxima da publicação original). Determinístico e
  reprodutível.
- **Ordem "dedup antes de Opção 7" é conceitualmente mais limpa** e não
  altera materialmente o resultado — das 597 duplicatas, a maioria
  (~424) está em classes grandes (`colunas+tv+mercado+tec+ilustrissima`)
  que sobrevivem a qualquer threshold razoável. Das categorias que ficam
  em `outros` na Opção 7, o impacto é ainda menor.
- **Coluna `eh_duplicata_text` segue a convenção existente** do projeto
  (`eh_mercado` em `src/eda/carregamento.py:35`) e é virtualmente
  gratuita (uma coluna booleana).

Contrapontos registrados para consciência (reconhecidos e aceitos):

- **Preservar near-dups deixa risco material de leakage entre splits**
  no ciclo futuro de splitting. Exemplo: `colunas↔colunas=1.127` pares;
  num split aleatório 80/20, a probabilidade de pelo menos um par irmão
  cair em treino+teste é essencialmente 1. Isso inflaria métrica da
  classe `colunas` e, por correlação, o reporte primário binário
  projetado (ADR 006) se `colunas` for sistematicamente predita como
  `não-mercado`. Mitigação: o ciclo de splitting deve adotar **split
  group-aware** (id de grupo = cluster de near-dup) para manter pares
  irmãos no mesmo lado do split. Dependência registrada em §Consequências.
- **Preservar `title` exato em `ilustrada` e `colunas`** (1.929 linhas
  combinadas) deixa o grosso do problema de duplicação editorial
  intocado. Política consciente: a ADR 004 já marcou `colunas` e
  `ilustrada` como "duplicatas editoriais persistentes" no desenho Opção 7;
  esta ADR não resolve esse problema nem pretende resolver. Eventual
  análise adicional (ex.: remover `title` duplicados apenas dentro de
  `ilustrada` e `colunas`) fica como ciclo candidato futuro.
- **A política pode ser contestada em revisão BRACIS** se o revisor
  considerar que `title`-dup também é artefato. Defesa: reportar no
  artigo, com amostragem qualitativa, que `title`-dups correspondem em
  alta proporção a conteúdo serial legítimo; oferecer ablation "com vs.
  sem dedup de `title` exato" como análise de sensibilidade se houver
  folga.

## Consequências

### Habilita

- **Corpus raw reduzido para 166.456 artigos** (167.053 − 597). Opção 7
  aplicada sobre esse corpus saneado resulta em aproximadamente
  160.218 artigos (ajuste marginal sobre o valor declarado na ADR 004).
- **Coluna `eh_duplicata_text`** disponível em todos os artefatos de
  dados, permitindo:
  - Auditoria de quantas linhas canônicas representam grupos de dup.
  - Amostragem qualitativa para reportar no artigo (quantos pares,
    quantas categorias, exemplos).
  - Análise de sensibilidade: medir se excluir as canônicas (`eh_duplicata_text == True`) altera materialmente os resultados.
- **Análise A1 (FPs por `y_original`, ADR 004)** fica livre de
  contagem dupla do mesmo `text` em categorias diferentes — caso
  envolvia 39 pares cross-classe com `mercado` em contrapartes `tv`,
  `tec`, `poder`. Sem essa limpeza, FPs de `mercado` viriam inflados
  por conteúdo idêntico reclassificado.
- **Resolução de um componente do ciclo de deduplicação.** A política de
  exato-`text` está fechada; ciclos futuros sobre near-dups e title-dups
  (se forem abertos) operam sobre corpus já parcialmente saneado.

### Não resolve

- **Near-duplicates intra-classe** (`colunas=1.127, tec=931, mercado=178,
  tv=146` pares). Permanecem. Decisão explícita de manter.
- **Duplicatas exatas de `title`** (4.015 linhas). Permanecem.
- **Duplicatas editoriais estruturais** de `colunas` e `ilustrada`
  persistem como característica do corpus Opção 7.

### Dependências com outras ADRs

- **Condiciona o ciclo futuro de splitting** (ADR a ser numerada como 008):
  o desenho do split **deve ser group-aware** para evitar leakage por
  near-dups preservados. Alternativas: (i) identificador de cluster
  MinHash como chave de grupo; (ii) split temporal (que naturalmente
  separa near-dups se publicados em janelas distintas); (iii) declarar
  explicitamente que o risco é tolerado e reportar intervalo de confiança
  mais conservador. **A escolha é do ciclo de splitting, mas a restrição
  é herdada desta ADR.**
- **Complementa a ADR 004:** o corpus operacional Opção 7 passa a ser
  definido como "raw deduplicado por `text` exato + threshold ≥1.000 +
  relabel top-6 concorrentes + mercado + outros". A ADR 004 §D.5
  reportava 160.815 sob a contagem pré-dedup; o valor pós-dedup é
  aproximadamente 160.218. A diferença é marginal e não altera a lógica
  de nenhuma decisão em 004.
- **Não afeta ADR 005** (validação externa WikiNews, em `proposta`,
  postergada). Quando ciclo próprio for aberto, avaliar se dedup
  equivalente é aplicada ao WikiNews (presumivelmente sim, por simetria
  metodológica).
- **Não afeta ADR 006** (métrica primária e setup de treino
  multiclasse/binário projetado). A métrica opera sobre o mesmo corpus
  pós-dedup.

### Revisão futura esperada

- **Sensibilidade a near-dups no split.** Se o ciclo de splitting
  adotar group-aware e isso mostrar gap material entre split aleatório
  e split group-aware (ex.: F1 aleatório > F1 group-aware por >2 p.p.),
  reportar no artigo como evidência concreta do risco de leakage e
  defender group-aware como escolha primária.
- **Possível ADR futura sobre near-dups.** Se durante a execução for
  observado comportamento anômalo atribuível a near-dups (ex.: modelo
  "memoriza" colunistas específicos), abrir ciclo específico para
  decidir política de remoção.
- **Possível ADR futura sobre `title`-dup em `ilustrada`/`colunas`.**
  Mesmo gatilho — se análise de FPs sugerir que `title`-dup contamina
  o diagnóstico de `colunas` e `ilustrada`, abrir ciclo específico.

### Pendências de implementação

- Adicionar coluna `eh_duplicata_text` em `src/eda/carregamento.py` (ou
  módulo de preprocessamento sucessor, se houver refatoração) seguindo
  a convenção já estabelecida por `eh_mercado`.
- Documentar o desempate (`ordem: (text, date, link) ascendente; manter
  primeiro`) no código, com teste unitário cobrindo um caso de
  `date`-empate resolvido por `link`.
- Reportar no artigo, na seção de metodologia: (i) política adotada,
  (ii) volume removido, (iii) exemplos amostrados, (iv) restrições
  explícitas (near-dups e `title`-dups preservados) com justificativa.

## Revisão em 2026-04-24

Ajustes decorrentes da materialização da política no código (Fase 1
do ciclo de implementação).

1. **Localização final dos módulos.** A pendência previa adicionar a
   coluna `eh_duplicata_text` em `src/eda/carregamento.py` "ou módulo
   de preprocessamento sucessor". Escolha final: **módulo sucessor**.
   Lógica de carregamento+dedup fica em `src/preprocessamento/` (em vez
   de misturar com a EDA, que permanece como leitura estática do
   corpus raw):
   - `src/preprocessamento/carregamento.py:preparar_corpus` adiciona
     `id_raw` e filtra nulos.
   - `src/preprocessamento/deduplicacao.py:deduplicar_por_texto_exato`
     aplica o desempate `(text, date, link)` via `mergesort` estável
     e produz a flag `eh_duplicata_text`.
   - Teste unitário de empate por `link` coberto em
     `tests/preprocessamento/test_deduplicacao.py:test_desempate_por_link_lexicografico_em_date_igual`.
   O arquivo `src/eda/carregamento.py` não foi alterado — continua
   servindo a EDA existente sem modificação.

2. **Números definitivos pós-pipeline e reconciliação do "597".** A
   execução de `scripts/preprocessar.py` sobre o `articles.csv` atual
   produziu:

   | Quantidade | Valor |
   |---|---:|
   | Corpus bruto (pré-filtro de nulos) | 167.053 |
   | Pós-filtro de `date`/`text` nulos | 166.288 |
   | Linhas efetivamente removidas pela dedup | 387 |
   | Sobreviventes marcados com `eh_duplicata_text=True` | 210 |
   | Corpus pós-dedup (`corpus_opcao*.parquet`) | 165.901 |

   A reconciliação com o **"597 duplicatas exatas por `text`"**
   declarado no §Contexto desta ADR: **210 grupos** de `text`
   repetido existem no corpus raw; cada grupo tem pelo menos 2
   linhas; sobrevivem 210 (uma canônica por grupo), removem-se 387.
   Total de linhas "envolvidas em duplicação" = 210 + 387 = **597**,
   exatamente o número reportado na §Contexto. A EDA original
   contabilizou "linhas envolvidas"; o pipeline reporta "linhas
   removidas". Ambos os números coexistem sem conflito — a flag
   `eh_duplicata_text` permite reproduzir qualquer das duas vistas.

3. **Previsão em §Consequências ajustada.** §"Habilita" previa
   "corpus raw reduzido para 166.456 artigos (167.053 − 597)"; o valor
   obtido é **165.901** (167.053 − 765 filtro de nulos − 387 dedup).
   Diferença de −555 (~0,33%) atribuída ao filtro de `date`/`text`
   nulos (pré-requisito da ADR 008 para split temporal), não previsto
   explicitamente nesta ADR. A alteração é neutra do ponto de vista
   metodológico: preservar linhas com `date` nulo não seria viável
   no protocolo de rolling-origin em qualquer caso.

4. **Ordem de operações preservada.** A §Decisão §2 fixou "dedup
   aplicada sobre o corpus raw antes do threshold Opção 7 e antes do
   relabel". `scripts/preprocessar.py` cumpre essa ordem: (i)
   carregamento+filtro de nulos; (ii) dedup; (iii) recortes (Opção 7,
   4, 3). A enumeração da Opção 4 também opera sobre o corpus
   pós-dedup, garantindo que o threshold ≥500 exclua categorias após
   a remoção das duplicatas.

5. **Restrição para splitting já consumida.** §Consequências previa
   que o ciclo futuro de splitting adotasse estratégia *group-aware*
   para mitigar leakage por near-dups preservados. O ciclo correu
   como ADR 008 e escolheu **não** agrupar: rolling-origin temporal
   primário + k-fold estratificado sem agrupamento como ablation
   intencional, com o leakage medido e **reportado como diagnóstico**
   (não mitigado). Medição empírica pós-pipeline em `src/splitting/leakage.py`
   confirma o sinal esperado: **rolling 4,2 %** vs. **k-fold 32,6 %**
   de pares near-dup cruzando treino↔teste. Esse contraste de ~8×
   é um dos resultados metodológicos centrais do artigo.

6. **Diagnóstico de leakage dependente de `id_raw`.** Para mapear
   os `id_a`/`id_b` de `artifacts/eda/tabelas/07-near-duplicates-pares.csv`
   (que são índices no DataFrame pós-ordenação pré-filtro da EDA) ao
   corpus pós-dedup, adicionou-se a coluna `id_raw` em
   `preparar_corpus`. Essa coluna não estava prevista nesta ADR; é
   adição colateral registrada também em R.5 da ADR 009.
