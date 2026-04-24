---
numero: 002
slug: plano-eda
data: 2026-04-23
ciclo: eda-inicial
status: aceita
depende_de: [001]
---

# 002 — Plano de EDA: caracterização da classe "mercado" ao longo do tempo

## Contexto

Antes de escolher estratégia de validação (split temporal puro, *stratified
K-Fold* aleatório, ou variantes híbridas como *rolling-origin* / *blocked CV*),
precisamos caracterizar **como a classe `mercado` se comporta no período
coberto pelo corpus FolhaUOL**. Esta EDA **não decide o split** — produz as
evidências que o ciclo seguinte vai consumir: há deriva de prevalência? há
sazonalidade? vocabulário drifa? duplicatas contaminariam qualquer split?

**Schema confirmado** (inspeção leve no planejamento):
- Colunas: `title, text, date, category, subcategory, link`.
- `category` é single-label; classe positiva = `category == "mercado"` após
  normalização.
- `date` em formato `YYYY-MM-DD` (granularidade diária).
- Prevalência em amostra ~13%, compatível com os ~12,6% globais declarados.

## Alternativas consideradas

| Dimensão | Opções | Escolha |
|---|---|---|
| Escopo de eixos | (a) mínima (só série) / (b) intermediária / (c) ampla | **(c) ampla** — descobrir contaminação tarde no pipeline é caro |
| Granularidade principal | diária / semanal / mensal | **mensal** (principal); diária/semanal no apêndice |
| Teste de deriva | Z de proporções / Qui² k×2 / Mann-Kendall | **Mann-Kendall primário**; Z e Qui² secundários |
| Near-duplicates | ciclo separado / nesta EDA | **nesta EDA** (MinHash/LSH) |
| Cauda de categorias | detalhar todas / agregar cauda | **top-7 + "outras"** |
| Figuras | PNG / SVG / ambos | **PNG + SVG** (PNG 300 dpi para review, SVG para artigo) |
| Período de corte | filtrar 2015–2017 estrito / tudo no CSV | **tudo que estiver no CSV** — a EDA reporta o que existe |
| Normalização do alvo | literal `"mercado"` / `lower + strip accents` | **lower + strip accents**; `subcategory` ignorado |

## Decisão

### Definição operacional do alvo

```
normalizar(s) := strip_accents(lower(strip(s)))
eh_mercado(row) := normalizar(row.category) == "mercado"
```

`subcategory` é ignorada. Linhas com `category` nula são excluídas do alvo e
tabuladas separadamente em §1.

### Eixos de análise

1. **Volume e prevalência global.** Total de linhas; nulos por coluna;
   `value_counts(category)` aplicando política top-7 (ver abaixo); prevalência
   global de `mercado` e IC 95%.
2. **Série temporal da classe.** Agregação **mensal** de total, positivos e
   prevalência. Diária e semanal no apêndice. Eixo X calendário completo
   mesmo com meses zerados.
3. **Testes de deriva.**
   - **Primário:** Mann-Kendall sobre a série mensal de prevalência.
   - **Secundários:** Z de proporções entre blocos (ex.: 2015 vs. 2017);
     Qui² k×2 com k = anos disponíveis.
   Reportar magnitude (diferença absoluta, risco relativo), não só p-valor.
3. **Sazonalidade.** Prevalência por dia-da-semana; heatmap mês×ano.
4. **Lacunas e anomalias.** Dias com contagem zero; dias com
   `|z|>3` sobre a série diária.
5. **Co-ocorrência lexical** entre `mercado` e as **top-7 demais** categorias.
   Sobreposição de tokens via TF-IDF por categoria (sem cherry-picking de
   vocabulário). Cauda de categorias agregada como `outras`.
6. **Duplicatas e near-duplicates.**
   - **Exatas:** duplicidade de `link`, de `title`, de `text`.
   - **Near-duplicates:** MinHash + LSH sobre shingles de `text`. Threshold
     inicial sugerido: Jaccard ≥ 0,85 (calibrável em amostra durante
     implementação).
   Se houver pares em **períodos distintos**, listar — contamina split
   temporal e aleatório.
7. **Deriva mínima de vocabulário.** Por trimestre, sobre notícias `mercado`:
   comprimento médio de `text`/`title`; Jaccard entre top-50 tokens de
   trimestres consecutivos; PSI sobre bag-of-top-200-tokens.

### Política "top-7 + outras"

Qualquer visualização ou tabela que enumere categorias detalha as 7 maiores
por contagem total no dataset completo; restante agregado em `outras`.
Aplicável a §1 (listagem geral) e §5 (co-ocorrência). Não se aplica aos eixos
que operam apenas sobre a classe positiva (§2, §3, §4, §7).

### Arquitetura

- **Módulos puros em `src/eda/`:**
  - `carregamento.py` — `carregar_articles(caminho_csv) -> DataFrame` com
    `date` tipada (`datetime64[ns]`), `category` normalizada (lower + strip
    accents), ordenação estável por `(date, link)` com `kind='mergesort'`.
  - `contagens.py` — `prevalencia_por_periodo`, `serie_total_e_classe`.
  - `testes_temporais.py` — wrappers para Mann-Kendall, Z de proporções, Qui².
    Retornam `dataclass` com nome, estatística, p-valor, efeito.
  - `duplicatas.py` — detectores exatos e MinHash/LSH.
  - `vocabulario.py` — tokenização simples, top-k, Jaccard, PSI.
  - `figuras.py` — funções puras que devolvem `matplotlib.figure.Figure`.
    Sem `plt.show()` dentro.
- **Notebook fino:** `notebooks/eda/01-caracterizacao-classe-mercado.ipynb`.
  Só importa de `src/eda/`, executa e exibe.
- **Seed global:** `SEED = 20260423`.
- **Saídas:**
  - Tabelas: `artifacts/eda/tabelas/*.csv` (UTF-8).
  - Figuras: `artifacts/eda/figuras/*.png` (300 dpi) **e** `*.svg`.
  - Nomes com prefixo numérico alinhado aos eixos (`01-*`, `02-*`, ...).

### Paradas de discussão durante execução

- **Após §1** — se volume ou nulos surpreenderem (ex.: total ≠ ~167k,
  prevalência fora de 10–15%, >1% de nulos em `category`/`date`).
- **Após §6** — se duplicatas (exatas + near) > 1% do corpus. Muda todas as
  contagens subsequentes e exige política de deduplicação antes de continuar.
- **Antes de fechar §3** — para confirmar qual teste entra como figura
  principal do artigo.

## Justificativa

- Escopo amplo é barato em tempo e reduz o risco de descobrir contaminação
  (duplicatas, deriva, lacunas) tarde no pipeline — custo editorial alto para
  submissão BRACIS.
- Granularidade mensal suaviza ruído de calendário editorial sem esconder
  deriva estrutural; diária/semanal permanecem no apêndice para anomalias.
- Mann-Kendall é agnóstico à forma funcional da tendência e adequado a uma
  série mensal curta (≈36 pontos). `[CITAÇÃO PENDENTE: Mann 1945; Kendall 1975
  — teste não-paramétrico de tendência monotônica]`
- MinHash/LSH é padrão consagrado para *near-duplicates* em corpora de
  notícias. `[CITAÇÃO PENDENTE: Broder 1997 — MinHash sobre shingles]`
- Política top-7 + "outras" é decisão explícita do usuário para manter as
  visualizações legíveis; o corte em 7 é arbitrário mas registrado.
- Normalizar acentos+case no alvo elimina falso-negativo silencioso caso
  haja variantes de digitação (`Mercado`, `MERCADO`, etc.). Reportar quantas
  linhas foram capturadas pela normalização em §1.

## Consequências

**Habilita**
- Decisão informada no ciclo seguinte sobre split temporal vs. estratificado
  aleatório vs. rolling-origin.
- Corpus com duplicatas auditadas antes de qualquer modelagem — pré-requisito
  de validação honesta.
- Tabelas e figuras reutilizáveis no artigo (formato SVG).

**Pendente / ciclos futuros**
- **Escolha do split** é o ciclo imediatamente seguinte, consumindo as saídas
  desta EDA.
- **Deriva semântica fina** (embeddings, tópicos) fica fora do escopo. Pode
  virar ciclo próprio se §7 sinalizar deriva lexical relevante.
- **Política de deduplicação** (manter primeira ocorrência? descartar pares?)
  só é decidida se §6 detectar volume significativo.
- **Lista de stopwords PT-BR**: `nltk.corpus.stopwords('portuguese')` vs.
  lista curada local — micro-decisão na implementação de `vocabulario.py`.
- **Threshold do MinHash** (sugestão inicial 0,85) pode exigir calibração em
  amostra antes de rodar no corpus completo.

---

## Revisão em 2026-04-23

Ajustes após execução de §1–§5 e investigação forense sobre dias-zero:

- **Numeração canônica das seções.** A lista original numerava "3. Testes de
  deriva" e "3. Sazonalidade" com o mesmo número. A execução renumerou:
  §1 volume, §2 série temporal, §3 testes de deriva, §4 sazonalidade,
  §5 lacunas/anomalias, §6 co-ocorrência, §7 duplicatas, §8 deriva vocab.
  **Referências futuras devem usar a numeração do notebook** (que é
  `notebooks/eda/01-caracterizacao-classe-mercado.ipynb`). O texto acima
  fica congelado como registro da intenção original.
- **Cobertura temporal efetiva** do corpus ficou abaixo do estimado:
  **2015-01-01 a 2017-09-30** (33 meses), não 2015–2017 completo.
  Out/2017 tem apenas 1 dia (n=121) e é excluído da agregação mensal.
  Detalhes e decisão operacional em `003-cobertura-temporal.md`.
- **18 dias zero em 2017** (dias 11 e 12 de jan–set): lacuna intrínseca do
  dataset-fonte (Kaggle/marlesson), fora do escopo corrigir. Documentada no
  `003-cobertura-temporal.md` e prevista como nota de rodapé no artigo final.
