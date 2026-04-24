---
numero: 001
slug: aquisicao-dataset
data: 2026-04-23
ciclo: inicial
status: aceita
---

# 001 — Aquisição e versionamento do dataset bruto

## Contexto

Precisamos de acesso confiável ao dataset FolhaUOL
(`marlesson/news-of-the-site-folhauol`, Kaggle — ~167k notícias FolhaSP/UOL
entre 2015–2017) como entrada para todos os experimentos do projeto. A decisão
envolve três dimensões: (i) onde armazenar os bytes, (ii) como garantir
integridade entre máquinas/ambientes, (iii) como permitir reprodução pelo aluno,
pelo avaliador BRACIS e por leitores do artigo.

## Alternativas consideradas

| Opção | Custo | Benefício |
|---|---|---|
| **A. Google Drive do usuário, sem versionar bytes** | Dependente do Drive específico do aluno; precisa mount no Colab. | Alinhado ao `CLAUDE.md §5` (Colab-first); zero custo de storage em git. |
| **B. Git LFS no repositório** | GitHub free = 1 GB de storage + 1 GB/mês de banda; duplica bytes de dataset já público; exige remote git. | Repositório autossuficiente: `git clone` traz tudo. |
| **C. Script de validação por hash SHA256, bytes fora do repo** | Depende da disponibilidade contínua da fonte; exige passo manual de download. | Zero storage em git; reprodutibilidade garantida por hash fixado; compatível com qualquer hospedagem futura. |
| **D. DVC com remote no Drive** | Adiciona uma ferramenta ao stack. | Une C (hash no git) e A (bytes no Drive). |

## Decisão

**Opção C.** Os bytes do dataset **não são versionados**. O repositório carrega
apenas metadados suficientes para verificar integridade:

- `scripts/verificar_dataset.py` — calcula o SHA256 do arquivo em `data/raw/`,
  registra no primeiro uso (TOFU) e valida em execuções seguintes. Possui flag
  `--extrair` para descompactar o zip após validação.
- `data/raw/DATASET_HASH.txt` — hash SHA256 de referência, commitado no git.
- `data/raw/.gitkeep` — marcador do diretório.
- `.gitignore` — exclui `data/raw/*` com exceção dos dois arquivos acima.

O **download é manual**, via navegador, na página do Kaggle. Nenhuma credencial
da API do Kaggle é necessária.

Caminho canônico: `data/raw/archive.zip` (ajustável via `--arquivo`).

## Justificativa

- Consistente com `CLAUDE.md §5` (dados no ambiente do usuário; scripts
  parametrizados por caminho).
- Padrão comum em trabalhos com corpora públicos de médio porte: versionar
  metadados e hashes, não bytes. `[CITAÇÃO PENDENTE: princípios FAIR de dados;
  "Datasheets for Datasets" (Gebru et al., 2021) — confirmar DOI antes de citar]`
- Evita cota de Git LFS para bytes de um dataset já hospedado publicamente.
- TOFU é adequado aqui: o dataset é público e estável; o risco mitigado é
  corrupção/deriva de versão entre máquinas, não adversarial.
- Não exige credenciais Kaggle — qualquer avaliador consegue baixar via navegador.

## Consequências

**Habilita**
- Qualquer passo de preprocessamento futuro pode assumir `data/raw/articles.csv`
  (após `--extrair`) como entrada canônica.
- Reprodutibilidade: clone do repo + download manual + execução do script
  reconstrói o ambiente de dados.
- Decisão compatível com migração futura para opção **D** (DVC) se o projeto
  crescer em complexidade de dados.

**Pendente / ciclos futuros**
- **Transporte para Colab.** Preprocessamento local, execução de modelos no
  Colab (conforme decisão do usuário no ciclo atual). Estratégia de transferir
  dados preprocessados para o Drive será um ciclo separado.
- **Fallback de hospedagem.** Se o dataset sair do ar no Kaggle, precisamos de
  mirror (HuggingFace Datasets, Zenodo com DOI). Avaliar no momento oportuno.
- **Hash de arquivos extraídos.** Atualmente só o zip é hasheado. Se
  descobrirmos não-determinismo na extração (improvável para zip), estender a
  validação.
