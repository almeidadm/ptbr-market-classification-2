"""
Geradores de janelas para detecção de drift (ADRs 010 e 011).

Três tipos de janela:

- **Mensal** (`gerar_janelas_mensais`): janelas contíguas que coincidem
  com meses-calendário, alinhadas ao primeiro dia do mês inicial do
  corpus. É a granularidade primária dos blocos B1 (estatístico) e B2
  (semântico).
- **Bi-semanal** (`gerar_janelas_bisemanais`): janelas contíguas de 14
  dias começando no primeiro dia do mês inicial. Curiosidade/anexo;
  replica a granularidade do Wanderley et al. STIL 2025. Última janela
  parcial é descartada para manter tamanhos consistentes na figura.
- **Aleatorizada** (`gerar_particoes_aleatorizadas` e
  `gerar_repeticoes_aleatorizadas`): baseline de controle. Permuta o
  corpus inteiro e particiona nos mesmos tamanhos das janelas
  temporais, preservando "cada artigo em exatamente uma partição".

Determinismo: as funções temporais não usam aleatoriedade (dependem
apenas de `df["date"]`). As aleatorizadas recebem `seed` explícito;
`gerar_repeticoes_aleatorizadas` deriva as seeds como
`SEED + repeticao_id` conforme ADR 011 §D.5.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.config import (
    DATA_FIM_DRIFT_EXCLUSIVO,
    DATA_INICIO_DRIFT,
    DIAS_POR_JANELA_BISEMANAL,
    N_REPETICOES_DRIFT,
    SEED,
)


@dataclass(frozen=True)
class Janela:
    """
    Partição do corpus para análise de drift.

    `indices_no_corpus` são posições no DataFrame de entrada, na mesma
    convenção do `Fold` em `src.splitting.fold`. `inicio` e `fim` ficam
    como `None` em janelas aleatorizadas.
    """

    indice: int
    rotulo: str
    indices_no_corpus: np.ndarray
    inicio: pd.Timestamp | None
    fim: pd.Timestamp | None

    def resumir(self) -> dict:
        return {
            "indice": int(self.indice),
            "rotulo": self.rotulo,
            "n_artigos": int(len(self.indices_no_corpus)),
            "inicio": self.inicio.isoformat() if self.inicio is not None else None,
            "fim": self.fim.isoformat() if self.fim is not None else None,
        }


def _validar_corpus(df: pd.DataFrame) -> pd.Series:
    if "date" not in df.columns:
        raise ValueError("DataFrame precisa ter a coluna `date`.")
    datas = pd.to_datetime(df["date"])
    if datas.isna().any():
        raise ValueError("`date` não pode ter valores nulos ao gerar janelas.")
    return datas


def filtrar_periodo_efetivo(
    df: pd.DataFrame,
    data_inicio: str | pd.Timestamp = DATA_INICIO_DRIFT,
    data_fim_exclusivo: str | pd.Timestamp = DATA_FIM_DRIFT_EXCLUSIVO,
) -> pd.DataFrame:
    """
    Restringe o corpus à cobertura temporal efetiva da ADR 003 antes de
    qualquer geração de janelas. Resolve a R2 da ADR 011: filtra `2017-10`
    da iteração mensal e do baseline aleatorizado simultaneamente,
    garantindo simetria entre `time_ordered` e `randomized`.

    O intervalo é meio-aberto `[data_inicio, data_fim_exclusivo)`. Reset
    de índice é responsabilidade do caller (manter consistência com
    `filtrar_por_escopo` nos scripts de execução).
    """
    datas = _validar_corpus(df)
    inicio = pd.Timestamp(data_inicio)
    fim = pd.Timestamp(data_fim_exclusivo)
    mascara = (datas >= inicio) & (datas < fim)
    return df.loc[mascara].reset_index(drop=True)


def gerar_janelas_mensais(df: pd.DataFrame) -> list[Janela]:
    """
    Gera uma janela por mês-calendário coberto pelo corpus.

    Janelas vazias (meses sem notícias) são omitidas. O rótulo segue o
    formato `YYYY-MM`. Em corpora muito esparsos isso pode produzir
    descontinuidades; documentar no consumo.
    """
    datas = _validar_corpus(df)
    inicio_corpus = datas.min().to_period("M").to_timestamp()
    fim_corpus = datas.max().to_period("M").to_timestamp() + pd.DateOffset(months=1)

    janelas: list[Janela] = []
    indice = 0
    cursor = inicio_corpus
    while cursor < fim_corpus:
        proximo = cursor + pd.DateOffset(months=1)
        mascara = (datas >= cursor) & (datas < proximo)
        indices = np.flatnonzero(mascara.values)
        if len(indices) > 0:
            janelas.append(
                Janela(
                    indice=indice,
                    rotulo=cursor.strftime("%Y-%m"),
                    indices_no_corpus=indices,
                    inicio=cursor,
                    fim=proximo,
                )
            )
            indice += 1
        cursor = proximo

    if not janelas:
        raise ValueError("Corpus não produziu nenhuma janela mensal não-vazia.")
    return janelas


def gerar_janelas_bisemanais(
    df: pd.DataFrame,
    dias_por_janela: int = DIAS_POR_JANELA_BISEMANAL,
) -> list[Janela]:
    """
    Gera janelas contíguas de `dias_por_janela` dias começando no
    primeiro dia do mês inicial. A última janela é descartada se
    parcial (menos de `dias_por_janela` dias entre cursor e fim do
    corpus).
    """
    if dias_por_janela <= 0:
        raise ValueError("`dias_por_janela` deve ser positivo.")
    datas = _validar_corpus(df)
    inicio_corpus = datas.min().to_period("M").to_timestamp()
    fim_corpus = datas.max() + pd.Timedelta(days=1)

    janelas: list[Janela] = []
    indice = 0
    cursor = inicio_corpus
    passo = pd.Timedelta(days=dias_por_janela)
    while cursor + passo <= fim_corpus:
        proximo = cursor + passo
        mascara = (datas >= cursor) & (datas < proximo)
        indices = np.flatnonzero(mascara.values)
        if len(indices) > 0:
            janelas.append(
                Janela(
                    indice=indice,
                    rotulo=f"{cursor.strftime('%Y-%m-%d')}_{(proximo - pd.Timedelta(days=1)).strftime('%Y-%m-%d')}",
                    indices_no_corpus=indices,
                    inicio=cursor,
                    fim=proximo,
                )
            )
            indice += 1
        cursor = proximo

    if not janelas:
        raise ValueError(
            "Corpus não produziu nenhuma janela bi-semanal não-vazia."
        )
    return janelas


def gerar_particoes_aleatorizadas(
    n_total: int,
    tamanhos: list[int],
    seed: int,
) -> list[Janela]:
    """
    Permuta `np.arange(n_total)` e fatia em partições com os tamanhos
    especificados. Preserva a propriedade "cada índice aparece em no
    máximo uma partição"; exige `sum(tamanhos) <= n_total`.

    O rótulo segue o formato `rand_NNN` para distinguir de janelas
    temporais (`YYYY-MM` ou `YYYY-MM-DD_YYYY-MM-DD`).
    """
    if n_total <= 0:
        raise ValueError("`n_total` deve ser positivo.")
    if any(t < 0 for t in tamanhos):
        raise ValueError("Tamanhos não podem ser negativos.")
    soma = sum(tamanhos)
    if soma > n_total:
        raise ValueError(
            f"sum(tamanhos)={soma} excede n_total={n_total}; "
            f"baseline aleatorizado sem reposição é infactível."
        )

    rng = np.random.default_rng(seed)
    indices_permutados = rng.permutation(n_total)
    cortes = np.cumsum([0] + list(tamanhos))

    janelas: list[Janela] = []
    for i, (ini, fim) in enumerate(zip(cortes[:-1], cortes[1:])):
        janelas.append(
            Janela(
                indice=i,
                rotulo=f"rand_{i:03d}",
                indices_no_corpus=indices_permutados[ini:fim],
                inicio=None,
                fim=None,
            )
        )
    return janelas


def gerar_repeticoes_aleatorizadas(
    n_total: int,
    janelas_referencia: list[Janela],
    seed_global: int = SEED,
    n_repeticoes: int = N_REPETICOES_DRIFT,
) -> list[list[Janela]]:
    """
    Gera `n_repeticoes` execuções independentes do baseline aleatorizado,
    cada uma com seed `seed_global + repeticao_id` conforme ADR 011 §D.5.

    O tamanho de cada partição em uma repetição é o mesmo da janela
    temporal correspondente em `janelas_referencia`, garantindo
    comparabilidade direta entre as duas condições.
    """
    tamanhos = [len(j.indices_no_corpus) for j in janelas_referencia]
    return [
        gerar_particoes_aleatorizadas(
            n_total=n_total,
            tamanhos=tamanhos,
            seed=seed_global + repeticao_id,
        )
        for repeticao_id in range(n_repeticoes)
    ]
