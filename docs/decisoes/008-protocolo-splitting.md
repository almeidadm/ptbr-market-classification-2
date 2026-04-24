---
numero: 008
slug: protocolo-splitting
data: 2026-04-24
ciclo: pré-experimentação
status: aceita
---

# 008 — Protocolo de *splitting*: rolling-origin *expanding window* primário com *stratified k-fold* como ablation

## Contexto

Todo o desenho experimental a partir deste ciclo depende de como o corpus
é particionado em treino, validação e teste. As ADRs anteriores impõem
três conjuntos de restrições herdadas:

1. **Deriva composicional documentada** (ADR 003, §D.2 da ADR 004).
   Mann-Kendall significativo em 24 de 48 categorias; prevalência de
   `mercado` sobe de 10,6% (2015) para 13,1% (2017); `colunas` e
   `ilustrada` crescem, `esporte` colapsa em 2017. **Deriva é parte do
   que o artigo investiga**, não ruído a neutralizar.

2. ***Near-duplicates* preservados** (ADR 007). Pares intra-classe em
   `colunas=1.127, tec=931, mercado=178, tv=146` permanecem no corpus.
   A ADR 007 transferiu para este ciclo a responsabilidade de mitigar
   *leakage* de *near-dups* entre partições.

3. **Pergunta de pesquisa sobre deriva elevada a resultado**
   (resposta do aluno Q2 deste ciclo, 2026-04-24). O artigo deve
   **reportar o perfil F1(t) de `mercado` ao longo do tempo** como
   contribuição, não apenas adotar uma validação qualquer.

Adicionalmente, restrição operacional imposta pelo aluno (Q3 deste
ciclo): **nenhum *grouping* baseado em *clusters* estimados**
(MinHash, K-means, embeddings). Apenas chaves observáveis no corpus
(`date`, `link`, `categoria`) ou derivadas triviais (ano, mês) são
elegíveis como chave de partição. Isso elimina a ***stratified group
k-fold* com `cluster_minhash` como chave de grupo** como alternativa
— o *leakage* de *near-dups* só pode ser *atenuado* via proximidade
temporal e *reportado* como diagnóstico, não *eliminado*
estruturalmente.

Esta ADR fixa o protocolo primário de validação para todos os
experimentos do artigo e o único protocolo de ablation reportado em
tabela suplementar. Demais ablations de recorte (ADR 004 Opções 4 e 3)
e de setup (ADR 006 item 4, binário puro) herdam este protocolo salvo
especificação contrária no ciclo de *setup experimental* ainda a ser
aberto.

## Alternativas consideradas

| Opção | Primário | Drift como resultado | Custo relativo | *Group-aware* exigido | Status |
|---|---|---|---|---|---|
| **A.** *Stratified k-fold* com *grouping* MinHash | não | não | médio | sim | **Rejeitada** (Q3: sem *clusters* estimados). |
| **B.** *Temporal holdout* único | sim | parcial (1 ponto) | baixo | não (proximidade temporal) | Rejeitada: gera estimativa única sem IC natural e sem perfil F1(t). |
| **C.** *Rolling-origin expanding window* | sim | sim (N pontos) | alto | não (proximidade temporal) | **Adotada**. |
| **D.** *Rolling-origin fixed window* | sim | sim (N pontos) | alto | não | Rejeitada (Q5): ablation adicional sem ganho metodológico suficiente no escopo atual. |
| **E.** *Rolling-origin* + tuning de hiperparâmetros único no fold 0 | sim | sim | médio-alto | — | Rejeitada (Q3): tuning re-executado a cada fold. |
| **F.** *Stratified k-fold* sem *grouping* como ablation contrastiva | não | não | médio | — | **Adotada como ablation**, deliberadamente sem *grouping* para maximizar contraste. |

## Decisão

### Protocolo primário: *rolling-origin expanding window* mensal

- **Granularidade temporal:** mensal, usando a coluna `date` do corpus.
- **Janela inicial de treino:** 18 meses (janeiro/2015 a junho/2016).
- **Janela de teste por fold:** 3 meses consecutivos, disjuntos entre
  folds.
- **Número de folds:** 5 — cobrindo as janelas de teste:
  - Fold 0: jul/2016–set/2016.
  - Fold 1: out/2016–dez/2016.
  - Fold 2: jan/2017–mar/2017.
  - Fold 3: abr/2017–jun/2017.
  - Fold 4: jul/2017–set/2017.
- **Treino cresce a cada fold** (*expanding*): fold 0 treina com os 18
  meses iniciais; fold 4 treina com 30 meses (2015-01 a 2017-06). Reflete
  cenário de *retraining* em *deployment*.
- **Determinismo:** ordem temporal define o *split*; não há sorteio ou
  *seed* no nível do *split* em si.

### Busca de hiperparâmetros: por fold, via *inner temporal split*

- **Busca re-executada integralmente a cada um dos 5 folds.** Modelo não
  carrega hiperparâmetros de fold anterior.
- ***Inner split* temporal:** dentro do bloco de treino de cada fold,
  os **últimos 3 meses** viram *inner validation* e o restante vira
  *inner train*.
  - Fold 0 (treino jan/2015–jun/2016): *inner train* = jan/2015 a
    mar/2016, *inner val* = abr/2016 a jun/2016.
  - Folds 1-4: mesma lógica, aplicada ao treino expandido do fold.
- **Espaço de busca:** a ser fixado no ciclo de *setup experimental*
  (ainda a abrir). Esta ADR compromete-se apenas com o protocolo, não
  com o *grid* concreto.
- **Critério de seleção:** F1 de `mercado` na projeção binária (métrica
  primária da ADR 006) computado na *inner val* do fold.

### Ablation: *stratified k-fold* sem *group-awareness* (k=5)

- **Estratificação** por `y_colapsado` (8 classes da Opção 7 da ADR 004).
- **Nenhuma chave de *grouping*** — *near-dup* pairs caem em lados
  opostos do *split* com probabilidade esperada ~80%. **Isso é
  intencional**: a ablation mede o otimismo de um protocolo que ignora
  simultaneamente deriva e *leakage*.
- **Mesma política de tuning por fold** que o primário (inner split
  estratificado 80/20 dentro do fold de treino, sem *seed* na ordem
  temporal mas com *seed* fixa no embaralhamento).
- *Seed* global para reprodutibilidade da ablation: **2026**.

### Diagnóstico obrigatório: *leakage* de *near-duplicates*

Para cada protocolo (primário e ablation), computar e reportar no
artigo:

- Fração de pares *near-dup* da ADR 007 (2.382 pares totais intra-classe)
  que caem em lados opostos (treino ↔ teste) em pelo menos um fold.
- Expectativa ancorada no desenho:
  - Rolling-origin: baixo — *near-dups* de colunas diárias (`colunas`,
    `tec`) tendem a publicar em janelas próximas.
  - *Stratified k-fold*: ~80% por fold (k−1)/k.
- O contraste entre esses dois números entra como evidência empírica
  objetiva do custo do protocolo ingênuo.

### Reporte de resultados esperado no artigo

1. **Tabela principal** com F1/PR-AUC binária projetada de `mercado`:
   - Uma linha por família de modelo.
   - Uma coluna por fold de *rolling-origin* (5 colunas).
   - Coluna de média e desvio-padrão cross-fold.
2. **Figura central** — F1(t) de `mercado` com IC por *bootstrap*
   dentro de cada janela de teste. Eixo x = mês central da janela; eixo
   y = F1. Uma curva por família de modelo. **Deriva de F1 é aqui o
   resultado, não premissa.**
3. **Tabela de ablation** reportando F1 médio do *stratified k-fold*
   junto com o rolling-origin, evidenciando o *gap*.
4. **Tabela de diagnóstico de *leakage*** com os números de *near-dup
   crossings* por protocolo.

## Justificativa

1. **Rolling-origin é a validação honesta sob deriva documentada.** A
   ADR 003 e a D.2 da 004 estabelecem empiricamente que o corpus tem
   deriva composicional significativa. *Stratified k-fold* aleatório
   mede generalização a exemplos *i.i.d.*; o cenário real é extrapolação
   temporal. Reportar F1 como se fosse i.i.d. seria metodologicamente
   desonesto para BRACIS dada a evidência disponível.

2. **O reenquadramento do eixo do artigo (ADR 004) exige perfil
   temporal como resultado.** Se o artigo é sobre "classificação em
   presença de bucket residual heterogêneo: diagnóstico, mitigação e
   custo", reportar F1 ao longo do tempo é parte do diagnóstico — não
   apenas do protocolo de validação. Rolling-origin é o protocolo que
   produz esse resultado naturalmente.

3. **Janela inicial de 18 meses (Q1).** Garante N mínimo por classe
   suficiente para BERTimbau — aproximadamente 87k artigos no fold 0,
   dos quais ~11k de `mercado`. Deixa 15 meses residuais que, com
   janelas de teste de 3 meses, produzem 5 folds — quantidade suficiente
   para caracterizar um perfil F1(t) com significância estatística
   razoável sem explodir custo computacional.

4. **Janela de teste de 3 meses (Q2).** Granularidade mais fina (1 mês)
   daria 15 folds mas com variância intra-fold alta (~5k artigos por
   teste); granularidade mais grossa (6 meses) daria só 2-3 folds,
   perdendo resolução do perfil. Três meses equilibra estabilidade
   intra-fold (~14-15k artigos por teste) com número de pontos no perfil
   (5 pontos).

5. ***Tuning* por fold via *inner temporal split* (Q3 e Q4).** O aluno
   optou pela alternativa mais rigorosa. Custo ~5× superior a tuning
   único, mas:
   - Evita o viés de hiperparâmetros fitados a uma única distribuição
     temporal que pode não generalizar pelos 5 folds.
   - Permite que o perfil F1(t) reflita comportamento realista de
     *retraining* (em *deployment*, tuning é típico a cada ciclo).
   - Produz diagnóstico secundário: se os hiperparâmetros ótimos
     mudam entre folds, é mais uma evidência de deriva.
   - *Inner split* temporal (e não estratificado aleatório) preserva o
     princípio "tudo temporal no primário"; usar estratificado
     aleatório introduziria inconsistência.

6. **Ablation *stratified k-fold* sem *grouping* (e não *group-aware*).**
   O ponto da ablation é contraste. Torná-la *group-aware* mascararia
   parte do *gap* que se quer medir — especificamente, a inflação de F1
   atribuível a *leakage*. Deliberado, não omissão. A ausência de
   *grouping* aqui é consistente com Q3 do aluno e com a ausência de
   *clusters* estimados em todo o desenho.

7. ***Near-dup leakage* reportado, não eliminado.** Consequência da Q3
   do aluno (sem *clusters* estimados) + ADR 007 (*near-dups*
   preservados). A honestidade metodológica exige reportar o resíduo
   como número, não silenciá-lo. A expectativa ancorada (baixo no
   rolling-origin por proximidade temporal) torna a posição defensável,
   mas o número vem dos dados.

Contrapontos registrados para consciência (reconhecidos e aceitos):

- **Custo computacional substantivo.** Para BERTimbau sobre o recorte
  primário (Opção 7): 5 folds × busca de hiperparâmetros por fold × N
  configurações no *grid* ≈ ordem de 15-25 *fine-tunings* só no primário.
  Os recortes de ablation (Opção 4 e Opção 3 da ADR 004) e o setup de
  ablation binária (ADR 006 item 4) multiplicam esse custo. O ciclo de
  *setup experimental* precisará decidir se todas as ablations adotam
  o mesmo protocolo de *tuning* por fold ou se se contentam com
  *tuning* único — esta ADR não prescreve essa decisão, apenas o
  protocolo primário.
- ***Leakage* residual não-nulo.** Mesmo com rolling-origin, *near-dups*
  com janela de publicação entre 2-3 anos de distância (p.ex. colunas
  republicadas em aniversários) cairão em lados opostos. A fração
  esperada é pequena mas não zero. Fica em diagnóstico.
- **Cinco pontos no perfil F1(t) é pouco para ajuste de tendência
  estatística robusta.** O artigo deve reportar o perfil como
  observação qualitativa + IC por ponto, não como regressão temporal
  com coeficiente publicável. Se deriva for direcional e monotônica,
  Mann-Kendall dos 5 pontos pode ser reportado como evidência fraca
  porém alinhada com a EDA.
- **Sazonalidade de 12 meses não é integralmente capturada** no perfil
  de 5 folds se a primeira janela de teste começa só em jul/2016. Para
  comparar "jul/2016" com "jul/2017" (mesmos meses do ano) temos só o
  fold 0 vs. fold 4. Isso fica como limitação explícita no artigo.

## Consequências

### Habilita

- **Protocolo de validação do artigo consolidado** para todos os
  experimentos primários e de ablation.
- **Perfil F1(t) como resultado central**, não apenas protocolo de
  validação.
- **Diagnóstico quantitativo de *leakage***, reforçando a honestidade
  metodológica.
- **Resolução parcial de um pré-requisito da ADR 005** (WikiNews OOD,
  em `proposta`): aquela ADR menciona dependência de "protocolo de
  *splitting* do FolhaUOL, para definir o que é *in-distribution*";
  com esta ADR o protocolo está definido. Demais pré-requisitos da 005
  continuam em aberto (obtenção do *dump*, mapeamento de taxonomia).
- **Resposta direta às perguntas iniciais do aluno sobre Stratified
  K-Fold vs. temporal split**: o artigo adota temporal como primário e
  estratificado como ablation contrastiva, com resultado quantificado.

### Impede ou torna custoso

- **Custo computacional cresce substancialmente** devido ao *tuning* por
  fold. O ciclo de *setup experimental* precisará dimensionar N
  configurações no *grid* de hiperparâmetros para caber no orçamento
  Colab.
- **Famílias de modelo com custo de *fine-tuning* muito alto** (ex.:
  modelos >300M parâmetros) podem precisar de desenho mais enxuto —
  menos configurações no *grid*, ou tuning por fold apenas para o
  modelo primário e *tuning* único para ablations.
- **Afirmar "validação cruzada de 5 folds"** sem qualificação temporal
  no artigo é inadequado — o protocolo não é k-fold canônico e
  confundiria revisor. Texto correto: "*rolling-origin expanding window*
  com 5 folds temporais consecutivos".

### Dependências com outras ADRs

- **Herda** restrição de preservação de *near-dups* da ADR 007 e opera
  dentro dela (não elimina *leakage*, reporta-o).
- **Herda** métrica primária (F1 e PR-AUC de `mercado` binária projetada)
  da ADR 006. A figura e tabelas principais reportam exatamente essa
  métrica por fold.
- **Resolve parcialmente** o pré-requisito de "protocolo de *splitting*"
  listado na ADR 005.
- **Precede e condiciona** o ciclo de *setup experimental* (ainda a
  abrir): aquele ciclo decide *grid* de hiperparâmetros, famílias de
  modelo, orçamento por ablation, formato de artefatos de experimento.
- **Não altera** ADRs 001, 002, 003, 004 quanto ao conteúdo delas; as
  tabelas de composição de classes e prevalências da 004 continuam
  válidas dentro do corpus já deduplicado pela 007.

### Revisão futura esperada

- Se o diagnóstico de *leakage* de *near-dups* em rolling-origin mostrar
  fração significativa (ex.: >10% dos pares cruzando folds), revisitar
  a decisão de preservá-los (ADR 007) ou adotar política mitigadora ad
  hoc (ex.: remover *near-dups* cross-fold após o *split* definido).
- Se o custo computacional do tuning por fold for proibitivo no Colab
  real, revisitar Q3 e cair para tuning único no fold 0, aceitando
  custo metodológico menor em favor de viabilidade.
- Se 5 pontos no perfil F1(t) se mostrarem insuficientes para sustentar
  a narrativa de deriva, revisitar granularidade da janela de teste
  (ex.: mover para 1 mês ganhando resolução).

### Pendências de implementação

- Implementar função de geração de *folds* rolling-origin em
  `src/splitting/` (módulo novo) com contrato:
  - Entrada: DataFrame com coluna `date` (datetime), parâmetros
    `janela_inicial_meses=18`, `janela_teste_meses=3`, `n_folds=5`.
  - Saída: lista de 5 pares `(indices_treino, indices_teste)` +
    lista de 5 pares `(indices_inner_train, indices_inner_val)` para
    cada fold.
- Implementar *stratified k-fold* ablation com `seed=2026` e
  estratificação por `y_colapsado`.
- Implementar função de diagnóstico de *near-dup leakage*:
  - Entrada: pares de *near-dups* da saída MinHash da EDA +
    atribuição fold de cada artigo.
  - Saída: fração de pares cruzando folds, por protocolo.
- Documentar `seed=2026` como constante do projeto em local central
  (p.ex. `src/config.py`).
- Testes unitários cobrindo:
  - Determinismo do rolling-origin (ordem temporal produz mesmos folds
    em execuções repetidas).
  - Determinismo do stratified k-fold sob mesma *seed*.
  - Detecção correta de *near-dup crossings* em um caso sintético
    pequeno.
