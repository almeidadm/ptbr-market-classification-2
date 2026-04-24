---
numero: 006
slug: tarefa-treino-e-metrica-primaria
data: 2026-04-24
ciclo: pré-experimentação
status: aceita
---

# 006 — Tarefa de treino multiclasse com métrica primária binária projetada

## Contexto

A tarefa de classificação do rótulo *mercado* no corpus FolhaUOL admite pelo
menos três formulações distintas no treino:

1. **Binária pura** — `mercado` vs. `não-mercado`, sem distinguir internamente
   as categorias negativas.
2. **Multiclasse plena** — cada categoria é rótulo próprio, conforme definido
   pela ADR 004 (agrupamento de categorias, atualmente em `rascunho`).
3. **Multiclasse com reporte binário projetado** — treino multiclasse, mas
   avaliação primária feita sobre a projeção "mercado vs. resto" do argmax
   das predições.

Evidência empírica relatada pelo aluno (consistente com literatura de
classificação desbalanceada sobre corpora heterogêneos) indica que supervisão
multiclasse entrega F1 e PR-AUC superiores na distinção `mercado vs. resto`
quando comparada ao treino binário puro sobre este mesmo corpus. A hipótese
mecânica é que a classe negativa binária (`não-mercado`) é uma mistura de
distribuições substantivamente distintas — `política`, `poder`, `esporte`,
`ilustrada`, `cotidiano` etc. —, e colapsar essa estrutura em um único rótulo
joga fora o sinal contrastivo que o classificador usaria para posicionar a
fronteira de `mercado`.

Em paralelo, a pergunta de pesquisa declarada em CLAUDE.md §1 e nas
`perguntas_iniciais/pergunta.txt` é **sobre a classe *mercado* especificamente**,
não sobre classificação temática geral. Reportar macro-F1 multiclasse como
métrica primária diluiria o foco e reduziria a comparabilidade com Garcia et al.
(2024), que reportam F1 por classe em cenário multiclasse, e com a ADR 005
(validação externa no WikiNews PT), cujo mapeamento é explicitamente binário
(`Economia e Negócios` → positivo; resto → negativo).

A ADR 004 já antecipou, em sua §Contexto, que o regime multiclasse seria
avaliado via "*one-vs-rest* do score individual de `mercado` após o treino
multiclasse". Esta ADR 006 formaliza esse desenho como decisão consolidada,
fixa a métrica primária de reporte, e desacopla explicitamente *setup de
treino* de *forma de reporte* — distinção que em ciclo anterior havia sido
colapsada na recomendação "binário como primário".

A ADR 005 (`proposta`) lista, entre seus pré-requisitos para promoção a
`aceita`, a "definição da métrica primária do experimento". A presente ADR
destrava essa dependência.

## Alternativas consideradas

| Opção | Treino | Reporte primário | Benefício | Custo / Risco |
|---|---|---|---|---|
| **A. Binário puro** | binário (`mercado` vs. `não-mercado`) | F1 / PR-AUC binária | Narrativa unívoca; pipeline simples; sem ambiguidade sobre o que é "classe positiva". | Desempenho empiricamente inferior segundo indicação do aluno; impede análise estrutural de FPs (não há matriz de confusão entre categorias); descarta sinal contrastivo. |
| **B. Multiclasse pleno** | multiclasse (ADR 004) | macro-F1 / PR-AUC multiclasse | Aproveita sinal contrastivo; reporte multiclasse é o mais rico. | Dilui a pergunta de pesquisa; perde comparabilidade direta com ADR 005 (binária); macro-F1 agrega classes irrelevantes ao problema (ex.: `esporte`, `ilustrada`) e mascara desempenho na classe de interesse. |
| **C. Multiclasse com reporte binário projetado** *(esta proposta)* | multiclasse (ADR 004) | F1 / PR-AUC binária de `mercado vs. resto` projetada do argmax multiclasse; macro-F1 e matriz de confusão multiclasse como secundárias | Aproveita o ganho empírico de B; preserva a pergunta unívoca de A; compatível com ADR 005 sem ajuste; matriz de confusão multiclasse alimenta a análise semântica de FPs pedida pela orientadora. | Requer documentar explicitamente o protocolo de projeção (argmax, e opcionalmente threshold sobre `P(mercado)`) para evitar crítica de "reporte conveniente". Modelos sem probabilidade calibrada precisam de calibração para PR-AUC. |
| **D. Treinos paralelos (binário *e* multiclasse)** | ambos | F1 binária de cada um | Permite quantificar diretamente o gap multiclasse → binário. | Custo 2× de experimentação e hiperparâmetros; ambiguidade sobre qual resultado é o "oficial"; sobrescrito pela opção C + ablation pontual. |

As opções não são mutuamente exclusivas: C + uma linha de ablation binária
pura captura o benefício diagnóstico de D sem duplicar o regime de
experimentos.

## Decisão

**Adotar a opção C.**

1. **Setup de treino.** Todos os modelos do experimento principal são
   treinados em regime multiclasse, sobre o conjunto de classes definido
   pela ADR 004 quando promovida a `aceita`.
2. **Métrica primária.** F1 e PR-AUC de `mercado vs. resto`, calculadas
   sobre a projeção binária das predições multiclasse. A projeção padrão é
   argmax (classe predita = `mercado` → positivo; qualquer outra → negativo).
   Aplicável tanto ao split interno do FolhaUOL quanto à validação externa
   WikiNews (ADR 005).
3. **Métricas secundárias, reportadas sobre o split interno FolhaUOL.**
   Macro-F1 multiclasse e matriz de confusão entre categorias. Fornecem o
   substrato empírico para a análise de FPs semântica e para a discussão
   sobre quais categorias competem com `mercado` no espaço de representação.
4. **Ablation obrigatória.** Um treino binário puro (`mercado` vs.
   `não-mercado`) por família de modelo, reportado em tabela de ablation
   específica, para quantificar o custo de descartar o sinal contrastivo.
   Não substitui o resultado primário.
5. **Reporte complementar opcional.** Para modelos com `predict_proba`
   disponível, reportar também F1 de `mercado` obtido por threshold sobre
   `P(mercado)` (não argmax), em análise suplementar. Permite discutir
   calibração sem poluir o reporte primário.

## Justificativa

- **Empírica.** Supervisão multiclasse preserva a heterogeneidade interna da
  classe negativa, que em cenários com classe negativa composta por
  subdistribuições distintas tende a melhorar a discriminação da classe de
  interesse. A indicação do aluno sobre desempenho superior do multiclasse
  neste corpus é consistente com esse mecanismo (Japkowicz & Stephen, 2002,
  DOI [10.3233/IDA-2002-6504](https://doi.org/10.3233/IDA-2002-6504);
  He & Garcia, 2009, DOI [10.1109/TKDE.2008.239](https://doi.org/10.1109/TKDE.2008.239)).
- **Alinhamento com CLAUDE.md §1.** O projeto declara como contribuições
  desejáveis "análise de erros" e "medição de deriva". A matriz de confusão
  multiclasse é o artefato natural para a análise estrutural de FPs — sem
  ela, resta apenas inspeção caso a caso, sem padrão identificável.
- **Alinhamento com as recomendações da orientadora.**
  `recomendacoes_orientadora.txt` enfatiza "análise cuidadosa de falsos
  positivos". Em regime binário, um FP é opaco ("errou: não era mercado"); em
  regime multiclasse projetado, é atribuível ("errou: confundiu `mercado`
  com `poder`"), o que é a pergunta semântica de fato.
- **Comparabilidade com Garcia et al. (2024).** O paper reporta F1 por
  classe em setup multiclasse. O reporte primário binário projetado deste
  projeto é derivável da mesma pipeline de avaliação e comparável diretamente
  com a linha `mercado` (ou categoria WikiNews equivalente) de Garcia et al.
- **Compatibilidade com ADR 005.** O mapeamento binário WikiNews permanece
  válido sem ajuste.
- **Posicionamento para BRACIS.** A narrativa do artigo fica: "treinamos
  multiclasse para aproveitar sinal contrastivo; reportamos binário projetado
  como métrica primária porque a pergunta de pesquisa é sobre *mercado*
  especificamente; a matriz de confusão multiclasse alimenta a análise
  semântica de FPs; a ablation binária quantifica o custo de descartar esse
  sinal". Responde antecipadamente à objeção "por que multiclasse?".

Contrapontos registrados para consciência:

- Projeção binária via argmax **não é equivalente** a um classificador binário
  com threshold calibrado. Se o modelo multiclasse hesita entre `mercado` e
  outra classe (ex.: `economia`, se existir como rótulo separado), o argmax
  pode cair para qualquer lado sem refletir a probabilidade acumulada em
  `mercado`. O item 5 da §Decisão mitiga isso via reporte suplementar por
  threshold.
- PR-AUC binária projetada requer probabilidade de `mercado` — disponível em
  modelos com `predict_proba` nativo (Regressão Logística, BERT, SVM com
  Platt scaling). Modelos sem probabilidade direta precisam de calibração
  (Platt ou isotônica) ou ficam restritos à F1.
- A escolha de "multiclasse no treino" herda toda a complexidade da ADR 004.
  Um pivô futuro no recorte de classes altera o espaço de argmax e, em
  princípio, os resultados primários — mesmo com o reporte permanecendo
  binário projetado. Isso está registrado em §Dependências.

## Consequências

### O que esta decisão habilita

- Análise de falsos positivos semântica via matriz de confusão multiclasse,
  respondendo diretamente à pergunta da orientadora e às dúvidas iniciais
  do aluno.
- Ablation binária como evidência empírica registrada para o próprio setup,
  em vez de afirmação não-demonstrada.
- Comparabilidade direta com a ADR 005 sem necessidade de re-experimentação
  ou re-mapeamento.
- Preservação da pergunta de pesquisa unívoca no relato primário do artigo.
- Desbloqueio da ADR 005 para promoção de `proposta` para `aceita` assim
  que os demais pré-requisitos forem cumpridos.

### O que esta decisão impede ou torna custoso

- Um pivô futuro para "classificação multiclasse plena como foco primário
  do artigo" (macro-F1 como métrica principal) exigiria revisão desta ADR
  e possivelmente das ADR 004 e 005.
- Modelos sem `predict_proba` nativo ficam parcialmente inelegíveis para
  PR-AUC sem esforço adicional de calibração.

### Dependências com outras ADRs

- **Resolvida por** ADR 004 (promovida a `aceita` em 2026-04-24): o
  conjunto concreto de classes do multiclasse primário é de 8 classes
  (Opção 7: `poder, colunas, mercado, esporte, mundo, cotidiano,
  ilustrada, outros`). As ablations reportadas em tabela suplementar
  usam 21 classes (Opção 4) e 5 classes (Opção 3). A ablation binária
  declarada no item 4 da §Decisão desta ADR opera sobre os mesmos 160.815
  artigos do recorte primário, com rótulo colapsado para `mercado` vs.
  `não-mercado`.
- **Desbloqueia parcialmente** a ADR 005 (validação externa WikiNews): a
  definição da métrica primária, listada como pré-requisito na 005, está
  agora consolidada nesta ADR.
- **Revisa uma consequência declarada na ADR 005.** A ADR 005 §Consequências
  afirma que "reformulação da tarefa para multiclasse de 5 ou mais classes
  [...] tornaria o mapeamento WikiNews → *mercado binário* custoso de
  refazer". No desenho adotado aqui essa preocupação não se aplica: o
  reporte primário permanece binário projetado, e o mapeamento WikiNews
  permanece válido sem alteração. A ADR 005 deve receber uma seção de
  "Revisão em 2026-04-24" registrando esse ajuste de escopo.
- **Precede** a ADR ainda inexistente sobre protocolo de *splitting* do
  FolhaUOL: a métrica primária definida aqui se aplica aos splits que
  aquela ADR vier a fixar, sem precedência sobre o desenho temporal vs.
  estratificado.
