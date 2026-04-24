"""
Testes estatísticos para caracterizar deriva temporal da prevalência.

Primário: Mann-Kendall (tendência monotônica sobre a série mensal).
Secundários: z de proporções entre dois blocos; Qui² k×2 entre k blocos.

Sempre retornar `ResultadoTeste` com estatística, p-valor e um dicionário
`efeito` com as magnitudes relevantes — p-valor isolado é insuficiente
para decisão metodológica.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from scipy import stats


@dataclass(frozen=True)
class ResultadoTeste:
    nome: str
    estatistica: float
    p_valor: float
    efeito: dict = field(default_factory=dict)


def mann_kendall(serie_prevalencia: Sequence[float]) -> ResultadoTeste:
    """Teste de Mann-Kendall via `kendalltau` contra o índice temporal.

    A estatística `tau` de Kendall calculada entre `(0, 1, ..., n-1)` e a
    série é equivalente em sinal e p-valor ao S-statistic clássico de
    Mann-Kendall para detecção de tendência monotônica em séries sem
    ties no eixo temporal.

    Valores nulos (períodos sem observações ou sem positivos tratados
    como nulo pelo chamador) são removidos antes do teste.
    """
    valores = np.asarray(serie_prevalencia, dtype=float)
    mascara = ~np.isnan(valores)
    valores = valores[mascara]

    if valores.size < 3:
        return ResultadoTeste(
            nome="mann_kendall",
            estatistica=float("nan"),
            p_valor=float("nan"),
            efeito={"n": int(valores.size), "motivo": "amostra insuficiente"},
        )

    indices = np.arange(valores.size, dtype=float)
    resultado = stats.kendalltau(indices, valores, variant="b")

    return ResultadoTeste(
        nome="mann_kendall",
        estatistica=float(resultado.statistic),
        p_valor=float(resultado.pvalue),
        efeito={
            "n": int(valores.size),
            "primeiro": float(valores[0]),
            "ultimo": float(valores[-1]),
            "delta": float(valores[-1] - valores[0]),
        },
    )


def z_proporcoes(n1: int, k1: int, n2: int, k2: int) -> ResultadoTeste:
    """Teste z bicaudal para diferença de duas proporções independentes.

    Usa o erro-padrão sob a hipótese nula (proporção combinada). Reporta
    diferença absoluta e risco relativo como magnitudes de efeito.
    """
    if n1 <= 0 or n2 <= 0:
        raise ValueError("n1 e n2 devem ser positivos")
    if not (0 <= k1 <= n1 and 0 <= k2 <= n2):
        raise ValueError("k_i deve estar em [0, n_i]")

    p1 = k1 / n1
    p2 = k2 / n2
    p_combinada = (k1 + k2) / (n1 + n2)
    variancia = p_combinada * (1.0 - p_combinada) * (1.0 / n1 + 1.0 / n2)

    if variancia == 0.0:
        estatistica = 0.0
        p_valor = 1.0
    else:
        erro_padrao = float(np.sqrt(variancia))
        estatistica = (p1 - p2) / erro_padrao
        p_valor = float(2.0 * (1.0 - stats.norm.cdf(abs(estatistica))))

    risco_relativo = float("nan") if p2 == 0.0 else p1 / p2

    return ResultadoTeste(
        nome="z_proporcoes",
        estatistica=float(estatistica),
        p_valor=float(p_valor),
        efeito={
            "p1": float(p1),
            "p2": float(p2),
            "diferenca": float(p1 - p2),
            "risco_relativo": float(risco_relativo),
            "n1": int(n1),
            "n2": int(n2),
        },
    )


def qui_quadrado_k_por_2(tabela: Sequence[Sequence[int]]) -> ResultadoTeste:
    """Qui-quadrado de independência sobre tabela k×2 (positivos, negativos).

    A tabela de entrada é uma sequência de k linhas, cada uma com
    `[positivos, negativos]` do bloco (por exemplo, um ano). Reporta
    também as prevalências observadas por bloco em `efeito`.
    """
    arr = np.asarray(tabela, dtype=int)
    if arr.ndim != 2 or arr.shape[1] != 2 or arr.shape[0] < 2:
        raise ValueError("tabela deve ter shape (k, 2) com k >= 2")

    chi2, p_valor, graus_liberdade, _ = stats.chi2_contingency(arr)
    prevalencias = (arr[:, 0] / arr.sum(axis=1)).tolist()

    return ResultadoTeste(
        nome="qui_quadrado_k_por_2",
        estatistica=float(chi2),
        p_valor=float(p_valor),
        efeito={
            "graus_liberdade": int(graus_liberdade),
            "prevalencias_observadas": prevalencias,
            "k": int(arr.shape[0]),
            "n_total": int(arr.sum()),
        },
    )
