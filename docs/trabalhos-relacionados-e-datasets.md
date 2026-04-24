# Trabalhos relacionados e datasets candidatos

**Tipo:** levantamento bibliográfico **não-oficial**.
**Status:** insumo para ciclos futuros. **Não consolida decisão metodológica.**
**Data:** 2026-04-23.
**Ciclo:** pré-experimentação.

Este documento lista (i) trabalhos que utilizam o corpus **FolhaUOL**
(`marlesson/news-of-the-site-folhauol`, Kaggle) e (ii) datasets adicionais
candidatos a entrar no projeto — como baseline, transfer learning ou validação
externa. Serve como cardápio para discussão; a adoção de cada item requer
ciclo próprio e eventual ADR em `docs/decisoes/`.

Convenções adotadas:
- Todos os itens foram localizados via busca pública. Metadados conferidos no
  PDF quando disponível localmente; do contrário, sinalizados com
  `[CITAÇÃO PENDENTE: ...]`.
- Nenhuma citação foi fabricada. Quando a referência não pôde ser verificada
  com rigor de artigo, o item é marcado como pendente.

---

## 1. Trabalhos que utilizam o FolhaUOL

### 1.1 Garcia, Shiguihara & Berton (2024)

- **Título:** *Breaking news: Unveiling a new dataset for Portuguese news
  classification and comparative analysis of approaches*.
- **Venue:** PLOS One, janeiro de 2024.
  DOI: [10.1371/journal.pone.0296929](https://doi.org/10.1371/journal.pone.0296929).
- **Uso do FolhaUOL:** recorte de 5 classes após limpeza
  (*power* 22.022, *market* 20.970, *sport* 19.730, *world* 17.130,
  *everyday* 16.967 — total 96.819), contraposto ao corpus novo
  **WikiNews PT** proposto pelos autores e à **AG News PT**
  (tradução via LibreTranslate).
- **Modelos comparados:** representações BoW, TF-IDF, *embeddings* ×
  SVM, CNN, DJINN, BERT.
- **Achado principal:** BERT com os melhores resultados em FolhaUOL e
  WikiNews PT; CNN abaixo do esperado.
- **PDF local:** `docs/analise-previa/Garcia et al. - 2024 - Breaking news [...].pdf`.

### 1.2 Alcoforado et al. (2022) — ZeroBERTo

- **Título:** *ZeroBERTo: Leveraging Zero-Shot Text Classification by Topic
  Modeling*.
- **Autores:** Alexandre Alcoforado, Thomas Palmeira Ferraz, Rodrigo Gerber,
  Enzo Bustos, André Seidel Oliveira, Bruno Miguel Veloso, Fabio Levy
  Siqueira, Anna Helena Reali Costa (USP · Télécom Paris · Universidade
  Portucalense & INESC TEC).
- **Venue:** PROPOR 2022 — 15th International Conference on Computational
  Processing of Portuguese. arXiv: [2201.01337](https://arxiv.org/abs/2201.01337).
- **Uso do FolhaUOL:** dataset completo via Kaggle; tarefa **reformulada
  como zero-shot multiclasse** apoiada em *topic modeling*.
- **Comparação:** ZeroBERTo × XLM-R em 0SHOT-TC. ~12% de melhoria em
  label-aware weighted F1 e ~13× mais rápido que o baseline transformer.
- **Relevância para o projeto:** contraponto *low-resource / label-scarce*.
  **Comparação numérica direta não é válida** (tarefa distinta da nossa).
  Útil como eixo de eficiência (Green AI) ou para posicionar a
  abordagem supervisionada como limite superior de referência.
- **PDF local:** `docs/analise-previa/2201.01337v3.pdf`.

### 1.3 Santana, Oliveira & Nascimento (2022)

- **Título:** *Text Classification of News Using Transformer-based Models for
  Portuguese*.
- **Autores:** Isabel N. Santana, Raphael S. Oliveira, Erick G. S. Nascimento
  (SENAI CIMATEC, Salvador-BA · University of Surrey, Reino Unido).
- **Venue:** Journal of Systemics, Cybernetics and Informatics (JSCI),
  vol. 20, n. 5, p. 33–59, 2022.
  DOI: [10.54808/JSCI.20.05.33](https://doi.org/10.54808/JSCI.20.05.33).
- **Uso do FolhaUOL:** re-mapeamento de **19 para 10 categorias** via
  K-means + TF-IDF, motivação explícita de redução de desbalanceamento.
- **Modelos comparados:** BERTimbau *fine-tuned* × Word2Vec (CBOW, Skip-gram)
  × Doc2Vec (via embedding de documento).
- **Métricas reportadas:** accuracy, weighted accuracy, precision, recall,
  F1-score, AUC ROC, AUC PRC.
- **Achado principal:** BERTimbau *fine-tuned* supera as demais
  representações em todas as métricas.
- **Ressalva de venue:** JSCI é periódico do IIIS, com rigor de revisão
  historicamente criticado. Usar como **referência de protocolo e antecedente
  técnico**, não como autoridade metodológica. Não citar como evidência
  isolada em revisão BRACIS.
- **PDF local:** `docs/analise-previa/SA702PC22.pdf`.

### 1.4 Implementações públicas (não são artigos)

- **Bert-Bertimbau (Zanara).** Repositório GitHub com fine-tuning de
  BERT/BERTimbau em dataset PT-BR derivado do FolhaUOL.
  [github.com/szanara/Bert-Bertimbau](https://github.com/szanara/Bert-Bertimbau).
  Utilidade: **referência de implementação**, não bibliográfica.

---

## 2. Datasets candidatos complementares

Organizados por proximidade à nossa tarefa (classificação binária da classe
*mercado* no FolhaUOL), em três faixas.

### 2.1 PT-BR, mesma tarefa (categorização temática de notícias)

#### WikiNews PT

- **Origem:** Garcia et al. (2024) — mesmo paper de §1.1. Coletado de
  [WikiNews em português](https://pt.wikinews.org).
- **Volume:** 9.135 notícias, distribuição **não balanceada** entre categorias.
- **Categorias relevantes:** inclui *Economia e Negócios* (tradução de
  *Economy and Business*), além de *Política*, *Cultura*, *Esportes* etc.
- **Licença:** CC-BY conforme WikiNews (confirmar na publicação dos autores).
  `[CITAÇÃO PENDENTE: link oficial de distribuição do recorte Garcia et al.]`
- **Uso plausível no projeto:** validação externa *out-of-distribution*
  (treino no FolhaUOL, teste na categoria análoga do WikiNews PT).
  Ver §3.

#### AG News PT (tradução)

- **Origem:** tradução do [AG News](https://www.kaggle.com/datasets/amananandrai/ag-news-classification-dataset)
  para o português via **LibreTranslate**, feita por Garcia et al. (2024).
- **Volume:** 120 mil treino / 7.600 teste. 4 classes balanceadas
  (*world, sports, business, sci/tech*).
- **Uso plausível:** sanity check de protocolo em cenário balanceado
  e com tradução automática. Cuidado: *business* no AG News é mais amplo
  que *mercado* no FolhaSP.

### 2.2 PT-BR, domínio próximo (textos econômicos/financeiros)

#### FinBERT-PT-BR (Leme)

- **Tipo:** modelo pré-treinado, não dataset rotulado para categorização.
- **Corpus de pré-treino:** ~1,4 milhão de notícias financeiras PT-BR.
  Corpus **não distribuído publicamente**.
- **Rotulação:** 500 sentenças anotadas com sentimento
  (positivo/negativo/neutro) para fine-tuning.
- **Checkpoint público:** [lucas-leme/FinBERT-PT-BR](https://huggingface.co/lucas-leme/FinBERT-PT-BR).
- **Uso plausível:** **continued pretraining / DAPT** — usar como
  inicialização alternativa a BERTimbau-base para fine-tuning na nossa
  tarefa binária. As labels de sentimento não são reaproveitáveis.

#### DANTEStocks

- **Título:** *DANTEStocks: A Multi-Layered Annotated Corpus of Stock Market
  Tweets for Brazilian Portuguese*.
  [Revista Brasileira de Linguística Aplicada](https://periodicos.ufmg.br/index.php/rbla/article/view/62521).
- **Volume:** 4.042 tweets, 80.997 tokens. Anotação em camadas:
  PoS + dependências (UD) · emoções (Plutchik) · NE (HAREM).
- **Uso plausível:** **contexto em Trabalhos Relacionados**, não experimento.
  Domínio (tweets curtos × matérias longas) e tarefas (NER/emoção ×
  categorização temática) são incompatíveis.
- **Trabalho derivado relevante:**
  *A Corpus of Stock Market Tweets Annotated with Named Entities*, PROPOR 2024.
  [aclanthology.org/2024.propor-1.28](https://aclanthology.org/2024.propor-1.28/).

#### B2T — Tweets sobre bancos brasileiros

- **Título aparente:** *B2T: A Dataset of Tweets in Portuguese Language about
  Brazilian Banks*.
  [ResearchGate 384899281](https://www.researchgate.net/publication/384899281).
- `[CITAÇÃO PENDENTE: autores, ano, venue — não verificados no PDF.]`
- **Uso plausível:** contexto apenas.

#### Previsão do mercado acionário brasileiro com BERT

- Capítulo Springer:
  [10.1007/978-3-031-21689-3_5](https://link.springer.com/chapter/10.1007/978-3-031-21689-3_5).
- `[CITAÇÃO PENDENTE: autores, ano exato, título completo — não verificados
  no PDF.]` Tarefa é **previsão de preço**, não categorização temática.
- **Uso plausível:** citação contextual em Trabalhos Relacionados.

### 2.3 EN / multilíngue (contexto e eventual pré-treino)

#### Financial PhraseBank (Malo et al.)

- **Volume:** ~4.840 sentenças EN de notícias financeiras, rotuladas
  sentimentalmente por 16 anotadores da área.
- **Distribuição:** [takala/financial_phrasebank](https://huggingface.co/datasets/takala/financial_phrasebank).
- `[CITAÇÃO PENDENTE: Malo et al., *Good debt or bad debt: Detecting semantic
  orientations in economic texts*, JASIST 2014 — confirmar título e DOI antes
  de citar no artigo.]`
- **Uso plausível:** contexto (gap de língua + gap de tarefa). **Não
  recomendado** para experimento direto.

#### FLUE / FLANG benchmark (Shah et al.)

- **Descrição:** benchmark financeiro EN com 5 tarefas, incluindo
  Financial PhraseBank.
- **Site:** [salt-nlp.github.io/FLANG](https://salt-nlp.github.io/FLANG/).
- `[CITAÇÃO PENDENTE: Shah et al., *When FLUE meets FLANG*, EMNLP 2022 —
  verificar.]`
- **Uso plausível:** referência de protocolo de avaliação em domínio
  financeiro.

#### Reuters RCV1 / RCV2

- Lewis, Yang, Rose & Li (2004), *RCV1: A New Benchmark Collection for Text
  Categorization Research*. JMLR, vol. 5.
  [Artigo](https://www.jmlr.org/papers/volume5/lewis04a/lewis04a.pdf).
- **Volume:** ~800k documentos Reuters anotados por categoria (RCV1, EN).
  RCV2 cobre 13 línguas, incluindo português parcial.
- **Uso plausível:** citação histórica em classificação multirrótulo /
  hierárquica de notícias. Não diretamente comparável.

#### MLSUM (Scialom et al., 2020)

- *MLSUM: The Multilingual Summarization Corpus*.
  [arXiv 2004.14900](https://arxiv.org/abs/2004.14900) · EMNLP 2020.
- **Importante:** cobre FR/DE/ES/RU/TR. **Não inclui português.**
- **Uso plausível:** nenhum direto para nós. Citado apenas para justificar
  ausência de análogo PT-BR no cenário multilíngue.

#### AG News (original, inglês)

- Coleção de ~1M notícias EN em 4 categorias.
- [Kaggle](https://www.kaggle.com/datasets/amananandrai/ag-news-classification-dataset).
- `[CITAÇÃO PENDENTE: Zhang, Zhao & LeCun, *Character-level Convolutional
  Networks for Text Classification*, NeurIPS 2015 — verificar.]`
- **Uso plausível:** só vale a versão traduzida (§2.1) no nosso contexto.

### 2.4 Modelos-fundação PT-BR (citação técnica obrigatória se adotados)

- **BERTimbau** — Souza, Nogueira & Lotufo (2020), BRACIS.
  [Springer 978-3-030-61377-8_28](https://link.springer.com/chapter/10.1007/978-3-030-61377-8_28).
  `[CITAÇÃO PENDENTE: confirmar página exata e ISBN antes de citar.]`
- **BERT models for Brazilian Portuguese: pretraining, evaluation and
  tokenization analysis** — Applied Soft Computing, 2023.
  [ScienceDirect S1568494623009195](https://www.sciencedirect.com/science/article/abs/pii/S1568494623009195).
  `[CITAÇÃO PENDENTE: autores completos.]`

---

## 3. Proposta não-oficial: WikiNews PT como validação externa OOD

> **Status:** proposta; **não é decisão**. Registrada para discussão em
> ciclo próprio antes de qualquer implementação.

### 3.1 Ideia

Após treinar nosso classificador binário de *mercado* no FolhaUOL,
**avaliar o mesmo modelo** — sem re-treino — no recorte de *Economia e
Negócios* do WikiNews PT (Garcia et al., 2024). A categoria funciona como
rótulo positivo OOD; as demais categorias do WikiNews PT agregam o lado
negativo.

### 3.2 Por que vale considerar

1. **Ataca diretamente uma pergunta declarada do projeto** em
   `docs/analise-previa/perguntas_iniciais/pergunta.txt`: como medir *data
   drift* e generalização fora da fonte de treino.
2. **Contribuição metodológica mais defensável perante BRACIS** do que F1
   marginal no próprio corpus. A comunidade valoriza honestidade sobre
   generalização.
3. **Viável com esforço limitado**: o WikiNews PT é público
   (via Garcia et al.) e menor que o FolhaUOL, cabe em memória Colab.
4. **Compatível com qualquer família de modelos** que venhamos a adotar
   (linear, transformer, zero-shot), ampliando o alcance da análise.

### 3.3 Riscos e objeções antecipadas

- **Mapeamento de taxonomia é interpretativo.** "Economia e Negócios" no
  WikiNews ≠ "Mercado" na Folha. É preciso documentar o mapeamento como
  decisão explícita antes do experimento, e reconhecer no artigo que a
  correspondência é aproximada. Sem esse cuidado, o resultado OOD vira
  crítica válida de revisor.
- **Tamanho do WikiNews PT é modesto** (~9k). Estimativas de F1 terão
  intervalo de confiança largo; será necessário reportar IC (bootstrap).
- **Possível overlap temporal / estilístico.** FolhaUOL é 2015–2017; o
  WikiNews PT do Garcia et al. tem janela temporal própria. `[CITAÇÃO
  PENDENTE: confirmar cobertura temporal do WikiNews PT em Garcia et al.]`

### 3.4 Pré-requisitos antes de promover a decisão

1. **Obter o recorte do WikiNews PT** distribuído por Garcia et al. e
   verificar licença/redistribuição.
2. **Documentar o mapeamento** de categorias WikiNews → rótulo binário
   *mercado/não-mercado* em ADR próprio antes de qualquer avaliação.
3. **Confirmar que avaliação OOD é eixo desejado** do artigo. Se o foco
   for puramente comparação de modelos dentro do FolhaUOL, esta proposta
   pode virar distração.

### 3.5 Alternativas caso esta proposta não prospere

- **Validação temporal interna** (split por data dentro do FolhaUOL) em vez
  de OOD por corpus. Menos ambicioso; responde parcialmente à mesma
  pergunta.
- **Replicação do setup do ZeroBERTo** (§1.2) no FolhaUOL para nosso
  classificador, reportando os mesmos números que o paper deles. Gera
  comparabilidade externa sem mudar o corpus.

---

## 4. Pendências de verificação

Itens marcados com `[CITAÇÃO PENDENTE: ...]` ao longo do documento exigem
conferência em PDF/DOI antes de qualquer uso em texto do artigo. Consolidação
destas pendências deve acontecer no ciclo em que cada item for de fato
citado — verificação especulativa antes disso é desperdício.

## 5. Histórico do levantamento

- **2026-04-23** — criação deste documento a partir de duas rodadas de
  busca em ciclo de discussão. Metadados de §1.1, §1.2 e §1.3 conferidos
  nos PDFs em `docs/analise-previa/`. Demais itens dependem de verificação
  futura.
