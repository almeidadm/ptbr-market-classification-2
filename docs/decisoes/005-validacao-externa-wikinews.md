---
numero: 005
slug: validacao-externa-wikinews
data: 2026-04-23
ciclo: pré-experimentação
status: proposta
---

# 005 — Validação externa *out-of-distribution* usando WikiNews PT

## Contexto

O corpus primário do projeto — FolhaUOL, ~167k notícias da FolhaSP/UOL entre
2015 e 2017 — é de fonte única e janela temporal estreita. Qualquer
classificador treinado nele corre o risco de capturar regularidades da folha
editorial, do vocabulário do período e do viés de seção da redação, em vez de
sinal semântico genuíno do que é "notícia de mercado". Essa é uma preocupação
explicitamente registrada pelo aluno em
`docs/analise-previa/perguntas_iniciais/pergunta.txt` (itens sobre *data
drift* e generalização), e também sinalizada nas recomendações da orientadora
em `docs/analise-previa/recomendacoes_orientadora.txt`.

Métricas *in-distribution* — F1, AUC-PRC e afins calculadas sobre *splits* do
próprio FolhaUOL, sejam eles estratificados, temporais ou por validação
cruzada — respondem à pergunta "o modelo aprendeu a distinguir exemplos deste
corpus?". Não respondem à pergunta "o modelo aprendeu o conceito de notícia
de mercado?". Para revisão BRACIS, onde honestidade sobre limitações pesa
tanto quanto ganho marginal de métrica, essa segunda pergunta é a
metodologicamente mais defensável.

O levantamento em
`docs/trabalhos-relacionados-e-datasets.md` identificou o **WikiNews PT**
(Garcia et al., 2024 — PLOS One, DOI
[10.1371/journal.pone.0296929](https://doi.org/10.1371/journal.pone.0296929))
como o único corpus público PT-BR de notícias jornalísticas, distinto da
Folha, que possui uma categoria semanticamente próxima ao rótulo *mercado*
(a categoria *Economia e Negócios*, tradução direta do esquema IPTC / WikiNews
original).

Esta ADR propõe adotar o WikiNews PT como **conjunto de validação externa
*out-of-distribution* (OOD)**, com protocolo a ser detalhado em ADR
sucessora caso a proposta seja aceita. O *status* aqui é `proposta`: os
pré-requisitos descritos em §Consequências precisam ser satisfeitos antes da
promoção a `aceita`.

## Alternativas consideradas

| Opção | Custo | Benefício | Risco residual |
|---|---|---|---|
| **A. WikiNews PT como validação externa OOD** (esta proposta) | Mapeamento taxonomia WikiNews → *mercado/não-mercado* é interpretativo e precisa ser documentado; tamanho do corpus (~9k) gera intervalos de confiança largos; eventual *overlap* temporal a confirmar. | Ataca diretamente a pergunta de generalização; contribuição metodológica forte para BRACIS; compatível com qualquer família de modelos (linear, transformer, zero-shot); corpus público e viável em Colab. | Correspondência semântica aproximada entre classes pode ser criticada por revisor se não explicitada. |
| **B. Apenas validação temporal interna no FolhaUOL** | Muito baixo — é um *split* adicional. | Responde parcialmente a *data drift* dentro da mesma fonte. Já é parte do plano-base (ADR 003 — cobertura temporal). | Não responde à pergunta de generalização entre fontes. Sozinha, é insuficiente para a contribuição metodológica do artigo. |
| **C. Replicação do setup ZeroBERTo (Alcoforado et al., 2022)** | Médio — exige reimplementar protocolo zero-shot do paper para comparação numérica. | Comparabilidade externa com um trabalho BRACIS/PROPOR citado, sem mudar de corpus. | Não resolve o problema OOD — continuamos avaliando só no FolhaUOL. Tarefa zero-shot é distinta da nossa. |
| **D. AG News PT (tradução automática, Garcia et al.)** | Baixo — corpus já traduzido e público. | *Sanity check* em cenário balanceado. | A categoria *business* do AG News é mais ampla que *mercado* na FolhaSP; tradução via LibreTranslate adiciona ruído; cenário balanceado não reflete o problema real. Mais útil como diagnóstico adicional do que como validação externa principal. |
| **E. Não fazer validação externa** | Zero. | Zero. | Perda da contribuição metodológica mais defensável do artigo. |

As opções não são mutuamente exclusivas. **A** e **B** são complementares:
A avalia generalização entre fontes, B avalia estabilidade ao longo do
tempo na mesma fonte. **D** pode entrar como diagnóstico auxiliar se houver
folga. **C** é ciclo autônomo de comparação com baseline, não de validação.

## Decisão (proposta)

**Adotar a opção A como eixo metodológico principal de validação externa**,
mantendo B (ADR 003) como avaliação temporal interna complementar. As
opções C e D ficam em reserva para ciclos futuros; a opção E é
explicitamente rejeitada.

Protocolo a ser detalhado em ADR sucessora quando esta for promovida:

1. **Origem dos dados.** Usar o recorte do WikiNews PT conforme distribuído
   pelos autores de Garcia et al. (2024) ou, se indisponível diretamente,
   recoletar a partir de `pt.wikinews.org` replicando o pipeline do paper.
2. **Mapeamento de rótulos.** Categoria *Economia e Negócios* → rótulo
   positivo (*mercado*); demais categorias → rótulo negativo. Casos de
   dupla rotulação e notícias ambíguas precisam de política explícita,
   registrada em ADR sucessora.
3. **Protocolo de uso.** Zero *retrain* ou *fine-tuning* no WikiNews — o
   corpus serve apenas como *test set* externo. Nenhum hiperparâmetro é
   ajustado olhando para o desempenho no WikiNews.
4. **Métricas reportadas.** As mesmas métricas primárias do experimento
   principal (a serem fixadas em ADR de métrica, ainda inexistente),
   acompanhadas de intervalo de confiança via *bootstrap* dado o tamanho
   modesto do corpus.
5. **Reportagem honesta de discrepâncias.** Se houver *gap* grande entre
   desempenho FolhaUOL e WikiNews, o artigo deve discutir causas plausíveis
   (taxonomia, estilo editorial, cobertura temporal) em vez de apresentar
   apenas o número melhor.

## Justificativa

- **Alinhamento com o CLAUDE.md §1**: a construção do artigo privilegia
  *"validação honesta"* e *"medição de deriva"* entre as contribuições
  metodológicas desejáveis. Validação externa OOD é a operacionalização
  direta dessas duas propriedades.
- **Alinhamento com `recomendacoes_orientadora.txt`**: a orientadora
  enfatiza comparação entre abordagens e análise cuidadosa de falsos
  positivos. Um conjunto externo expõe falsos positivos estruturais, não
  apenas os que decorrem do *split* do treino.
- **Posicionamento competitivo para BRACIS**: revisores valorizam evidência
  de generalização. F1 marginal maior num único corpus é crítica fácil
  ("*overfitting* à folha"). Relato honesto de queda/manutenção de
  desempenho em corpus externo é crítica muito mais difícil de fazer.
- **Custo proporcional ao benefício**: WikiNews PT é pequeno, público e já
  existe um pipeline conhecido para ele na literatura PT-BR (Garcia et al.).
  Não exige anotação nova, apenas mapeamento de taxonomia e avaliação.
- **Compatibilidade com o plano Colab-first** (CLAUDE.md §5): ~9k
  documentos cabem trivialmente em memória, inferência é rápida mesmo em
  T4.

Contrapontos registrados para consciência:

- A correspondência *Economia e Negócios* ↔ *mercado* não é perfeita.
  Quantos documentos do WikiNews cobrem o que a FolhaSP chamaria de
  "mercado" vs. o que chamaria de "economia" ou "poder" é uma pergunta
  empírica que só o mapeamento manual de amostra responde. Esse esforço
  precisa ser cotado no pré-requisito 1.
- O tamanho modesto do WikiNews PT torna diferenças pequenas entre modelos
  estatisticamente inconclusivas. A proposta é honesta sobre isso:
  reportamos IC, não reivindicamos *ranking* fino de modelos a partir desse
  corpus.

## Consequências

### Pré-requisitos antes de promover esta ADR a `status: aceita`

1. **Obtenção do recorte do WikiNews PT** (Garcia et al.) — verificar
   licença, redistribuição e reprodutibilidade do procedimento de coleta.
   Se os autores não distribuírem o *dump* diretamente, avaliar custo de
   recoletar.
2. **Mapeamento formal de taxonomia** WikiNews → binário *mercado*. Deve
   ser registrado em ADR sucessora, preferencialmente após uma rodada de
   inspeção manual em amostra aleatória (~100 documentos por categoria
   candidata) para validar a correspondência.
3. **Definição da métrica primária do experimento**. Esta ADR não fixa
   métrica; isso é ciclo autônomo. Precisa existir antes da avaliação real.
4. **Confirmação do eixo metodológico pelo usuário e, idealmente, pela
   orientadora.** Se o foco do artigo pender para comparação de famílias
   de modelos em vez de generalização entre fontes, esta proposta perde
   prioridade e pode ser reclassificada como análise complementar.

### O que esta proposta habilita se aceita

- Seção "Validação externa" autônoma no artigo, com contribuição
  metodológica independente dos resultados *in-distribution*.
- Resposta direta à pergunta sobre *data drift* registrada em
  `perguntas_iniciais/pergunta.txt`, citável no artigo.
- Comparabilidade parcial com Garcia et al. (2024), que já operou sobre
  ambos os corpora ainda que em setup diferente.

### O que esta proposta impede / torna custoso

- Reformulação da tarefa para multiclasse de 5 ou mais classes, caso seja
  desejada em ciclos futuros — o mapeamento WikiNews → *mercado binário*
  precisaria ser refeito e o experimento re-rodado.
- Uso do WikiNews também como **treino** ou **fonte de augmentação** —
  isso contaminaria o uso como validação externa e exige ADR explícita em
  sentido contrário se for desejado.

### Dependências explícitas com outras ADRs

- **Resolvida por** ADR 006 (métrica primária): F1 e PR-AUC de `mercado vs.
  resto` projetadas do argmax multiclasse. O protocolo de avaliação
  WikiNews, se esta ADR for promovida, reaproveita essa mesma métrica sem
  ajuste.
- **Depende de** ADR ainda inexistente sobre protocolo de *splitting* do
  FolhaUOL, para definir o que é *in-distribution* a ser comparado com OOD.
- **Complementa** ADR 003 (cobertura temporal) — as duas validações
  convivem sem conflito.

## Revisão em 2026-04-24

Ajustes decorrentes da consolidação da ADR 006 (tarefa de treino multiclasse
com métrica primária binária projetada) e de decisão de escopo do aluno
nesta mesma data:

1. **Pré-requisito de métrica primária — resolvido.** A lista em
   §Consequências / Pré-requisitos continha "Definição da métrica primária
   do experimento". A ADR 006 fixa essa métrica (F1 e PR-AUC binárias
   projetadas do multiclasse). Este item deixa de bloquear a promoção desta
   ADR a `aceita`.
2. **Consequência declarada que não se aplica ao desenho adotado.** Em
   §Consequências / "O que esta proposta impede", o texto original previa
   que "reformulação da tarefa para multiclasse de 5 ou mais classes [...]
   tornaria o mapeamento WikiNews → *mercado binário* custoso de refazer".
   No desenho consolidado na ADR 006 o reporte primário permanece binário
   projetado mesmo com treino multiclasse, de modo que o mapeamento
   WikiNews → binário permanece válido sem alteração. Essa preocupação
   fica registrada como não-aplicável a este recorte, preservada para o
   caso de pivô futuro para reporte multiclasse primário.
3. **Ordem de execução — WikiNews em segundo plano.** Decisão do aluno
   (2026-04-24): a avaliação no WikiNews PT é deliberadamente postergada
   para **depois** da fase de treinamento e avaliação no FolhaUOL. Até lá,
   os demais pré-requisitos desta ADR (obtenção do recorte WikiNews,
   mapeamento formal da taxonomia, confirmação da orientadora) ficam
   despriorizados. Esta ADR permanece com `status: proposta` e não será
   promovida a `aceita` até que a fase FolhaUOL esteja estabilizada e os
   pré-requisitos remanescentes sejam executados em ciclo próprio.
