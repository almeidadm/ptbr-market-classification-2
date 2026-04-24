---
numero: 003
slug: cobertura-temporal
data: 2026-04-23
ciclo: eda-inicial
status: aceita
depende_de: [001, 002]
---

# 003 — Cobertura temporal efetiva do corpus e tratamento de out/2017

## Contexto

Durante a execução da §5 do plano 002 (lacunas e anomalias) detectou-se padrão
sistemático de 18 dias com contagem zero, **todos em 2017**, exclusivamente
nos **dias 11 e 12 de cada mês de janeiro a setembro**. Hipótese inicial:
bug de parsing `YYYY-MM-DD` vs. `YYYY-DD-MM`, com notícias de nov–dez/2017
sendo lidas em datas erradas. A confirmação dessa hipótese mudaria toda a
narrativa temporal do artigo e exigiria rerrodar §1–§5.

## Investigação forense

Comandos diretos sobre `data/raw/articles.csv` (carregamento com
`pd.to_datetime(..., format="%Y-%m-%d", errors="coerce")`):

1. **Zero nulos após parse estrito.** Nenhum `errors="coerce"` foi acionado.
   O CSV já fornece datas `YYYY-MM-DD` válidas — não há ambiguidade de formato.
2. **Dias 11 e 12 em 2015 e 2016 têm contagens normais** (2.282 e 2.398 em
   2015; 1.700 e 1.697 em 2016), compatíveis com os vizinhos (dias 10 e 13).
   Se fosse bug de formato, afetaria os três anos.
3. **Dias 10 e 13 em 2017 têm contagens normais** (1.203 e 1.215). Se
   houvesse "absorção" de notícias dos dias 11/12 para os vizinhos, estes
   estariam inflados — não estão.
4. **Nov/2017 e dez/2017 não têm nenhum dia com dados.** Última data do
   corpus: `2017-10-01` (um único dia). Se notícias de nov–dez tivessem sido
   misparseadas, apareceriam em outras datas — desaparecem por completo.

**Conclusão:** a hipótese DD-MM é **falsa**. O dataset-fonte (Kaggle,
`marlesson/news-of-the-site-folhauol`) tem:

- **Truncamento em 2017-10-01** (coleta encerrou aí; out/2017 é 1 único dia).
- **Lacuna intrínseca nos dias 11 e 12 de jan–set/2017** (18 dias), provável
  artefato do scraper original. Fora do nosso escopo corrigir.

## Alternativas consideradas

| Opção | Trade-offs |
|---|---|
| A. Abandonar out/2017 (remover do corpus) | Perde 121 notícias sem ganho metodológico |
| B. Truncar corpus em 2017-09-30 | Mesma perda, sem ganho |
| C. Manter no corpus, excluir apenas da série mensal agregada | Mantém dados para modelagem; elimina o artefato gráfico |
| D. Tratar out/2017 como ponto mensal normal | Artefato visual (IC Wilson ~±6 p.p.) e estatístico (peso desproporcional em testes agregados por mês) |

## Decisão

**Opção C.**

- **Cobertura temporal declarada:** **2015-01-01 a 2017-09-30** — 33 meses
  completos contíguos.
- **Out/2017** (`2017-10-01`, n=121 notícias): **permanece no corpus bruto**
  para disponibilidade em treino/teste supervisionado futuro; **é excluído da
  agregação mensal** em §2 e §3.
- **Séries diária e semanal** permanecem sobre todo o intervalo; o gap de 18
  dias aparece nelas por natureza (contagens zero ou NaN em semanas afetadas).
- **18 dias zero** (dias 11 e 12 de jan–set/2017) ficam como limitação
  documentada do corpus-fonte, com nota de rodapé prevista no artigo final.

## Justificativa

- Out/2017 como ponto mensal viola o pressuposto de que cada ponto na série
  agregada represente um mês completo. Com n=121 vs. média ~5.000, a
  variância amostral dispara e o IC Wilson alarga ~6×, distorcendo tanto a
  figura quanto a narrativa.
- Mantê-lo no corpus preserva 121 notícias individuais para classificação
  supervisionada, onde cada observação conta por si (não agrega).
- Lacuna de 1,8% do calendário (18 / 1.005 dias) é pequena e documentável;
  não justifica redução adicional do corpus.
- Mann-Kendall sobre 33 pontos mantém poder adequado para detecção da
  tendência já observada (τ = 0,49, p < 1e-4).

## Consequências

**Habilita**
- Série mensal visualmente consistente e estatisticamente defensável.
- Figura `02-serie-mensal-prevalencia` sem o "trombetão" terminal, narrativa
  mais limpa para o artigo.
- Preservação do corpus completo (167.053 notícias) para modelagem.

**Requer re-execução**
- `02-serie-mensal.csv` e `02-cobertura-temporal.csv` refeitos sem out/2017.
- `02-serie-mensal-prevalencia.{png,svg}` e `02-serie-mensal-contagens.{png,svg}`
  refeitos sobre 33 pontos.
- `03-testes-deriva.csv` — Mann-Kendall recomputado sobre 33 pontos
  (alteração marginal esperada). Z e Qui² não afetados (agregam por ano).

**Pendente**
- Nota de rodapé no artigo final mencionando (i) truncamento em 2017-10-01
  e (ii) lacuna sistemática de 18 dias em 2017. TODO já registrado como
  célula markdown no notebook (§2).
- Se a estratégia de split futuro for temporal, a decisão de onde colocar
  out/2017 (treino, teste, ou excluído) é um sub-ciclo desse momento.

## Referência

Investigação forense completa: ver mensagens da sessão 2026-04-23 (blocos
de código Python executados interativamente, não persistidos como script —
o resultado é reprodutível em ~10 linhas sobre `data/raw/articles.csv`).
