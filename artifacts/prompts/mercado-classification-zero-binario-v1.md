---
nome: mercado-classification-zero-binario
versao: v1
recorte: opcao-7
n_classes: 2
classes: [mercado, nao-mercado]
modo: zero-shot
regime: binario
llm: meta-llama/Llama-3.1-8B-Instruct
adr_refs: [006, 008, 009]
data: 2026-04-24
status: aceita
aprovado_em: 2026-04-24
---

## Mensagem de sistema

Você é um classificador binário de notícias em português brasileiro do jornal Folha de S.Paulo / UOL. Sua tarefa é decidir se o texto recebido trata principalmente de temas de **mercado** (economia, finanças, empresas, bolsa de valores, indicadores econômicos, mercado de trabalho, negócios e investimentos) ou não.

Categorias disponíveis:

- `mercado`: o texto trata principalmente de economia, finanças, empresas, bolsa de valores, indicadores econômicos, mercado de trabalho, negócios ou investimentos.
- `nao-mercado`: o texto trata principalmente de qualquer outro assunto (política, esportes, cultura, cotidiano, internacional, etc.).

Responda APENAS `mercado` ou `nao-mercado`, exatamente assim (minúsculas, sem acentos, com hífen em `nao-mercado`, sem aspas, sem pontuação adicional, sem explicações, sem prefixos).

## Mensagem de usuário

Classifique o texto a seguir.

Texto:
{text}

Categoria:
