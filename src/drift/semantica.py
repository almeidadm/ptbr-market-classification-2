"""
Bloco B2 (análise semântica) da detecção de drift (ADRs 010 e 011).

Três métricas sobre embeddings já alinhados por janela:

- **Cosseno consecutivo** (`cosseno_centroide_consecutivo`): para cada
  par (i, i+1), distância de cosseno (1 − similaridade) entre os
  centróides das duas janelas. Mede deslocamento marginal mês-a-mês.
- **Cosseno cumulativo** (`cosseno_centroide_cumulativo`): para cada
  janela i ≥ 1, distância de cosseno entre seu centróide e a média dos
  centróides anteriores. Mede afastamento acumulado da história.
- **MMD² estatística** (`mmd2_consecutivo`): reusa `MMDDrift` do
  `alibi-detect` com `n_permutations=1` para extrair apenas o valor de
  `distance` (a estatística MMD² em si, com kernel RBF gaussiano e
  bandwidth via heurística da mediana). O p-value retornado pela
  permutação é descartado — o uso aqui é descritivo, e a variância
  amostral é capturada pelo baseline randomizado externo (ADR 011 §D.5).

Retorno uniforme em `ResultadoSemantica(metrica, janela_a, janela_b, valor)`.
Para o cosseno cumulativo, `janela_a` é o rótulo sintético
`historico_ate_<rotulo_anterior>` para tornar explícito que o ladrão é a
média histórica e não uma janela individual.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.config import SEED


@dataclass(frozen=True)
class ResultadoSemantica:
    metrica: str
    janela_a: str
    janela_b: str
    valor: float


def centroide(embeddings: np.ndarray) -> np.ndarray:
    """Média dos embeddings da janela em float64 para evitar acúmulo."""
    if embeddings.ndim != 2 or embeddings.shape[0] == 0:
        raise ValueError(
            "Centróide exige array 2D não-vazio (n, d); recebido "
            f"shape={embeddings.shape}."
        )
    return embeddings.astype(np.float64, copy=False).mean(axis=0)


def distancia_cosseno(a: np.ndarray, b: np.ndarray) -> float:
    """
    `1 − cos(a, b)`. Vetores com norma zero retornam `1.0` (interpretação
    conservadora: sem informação direcional, tratamos como ortogonal).
    """
    norma_a = float(np.linalg.norm(a))
    norma_b = float(np.linalg.norm(b))
    if norma_a == 0.0 or norma_b == 0.0:
        return 1.0
    similaridade = float(np.dot(a, b) / (norma_a * norma_b))
    return 1.0 - similaridade


def cosseno_centroide_consecutivo(
    embeddings_por_janela: list[np.ndarray],
    rotulos: list[str],
) -> list[ResultadoSemantica]:
    """
    Distância de cosseno entre centróides de janelas adjacentes (i, i+1).
    Produz `len(rotulos) − 1` resultados.
    """
    _validar_paridade(embeddings_por_janela, rotulos)
    centroides = [centroide(emb) for emb in embeddings_por_janela]
    resultados: list[ResultadoSemantica] = []
    for i in range(len(centroides) - 1):
        valor = distancia_cosseno(centroides[i], centroides[i + 1])
        resultados.append(
            ResultadoSemantica(
                metrica="cosine_centroid",
                janela_a=rotulos[i],
                janela_b=rotulos[i + 1],
                valor=valor,
            )
        )
    return resultados


def cosseno_centroide_cumulativo(
    embeddings_por_janela: list[np.ndarray],
    rotulos: list[str],
) -> list[ResultadoSemantica]:
    """
    Distância de cosseno entre a janela `i` e a média dos centróides de
    `[0..i−1]`. Produz `len(rotulos) − 1` resultados (a primeira janela
    não tem histórico).

    A "média histórica" aqui é a média dos centróides anteriores (não a
    média de todos os embeddings anteriores). Essa escolha pondera cada
    janela igualmente, evitando que meses com muitos artigos dominem o
    sinal — alinha com a comparação par-a-par já usada na consecutiva.
    """
    _validar_paridade(embeddings_por_janela, rotulos)
    centroides = [centroide(emb) for emb in embeddings_por_janela]
    resultados: list[ResultadoSemantica] = []
    for i in range(1, len(centroides)):
        historico = np.mean(np.stack(centroides[:i], axis=0), axis=0)
        valor = distancia_cosseno(historico, centroides[i])
        resultados.append(
            ResultadoSemantica(
                metrica="cumulative_cosine",
                janela_a=f"historico_ate_{rotulos[i - 1]}",
                janela_b=rotulos[i],
                valor=valor,
            )
        )
    return resultados


def mmd2_estatistica(
    x_a: np.ndarray,
    x_b: np.ndarray,
    *,
    seed: int = SEED,
    device: str | None = None,
) -> float:
    """
    Retorna apenas a estatística MMD² do `MMDDrift` (kernel RBF
    gaussiano, bandwidth via mediana). Usa `n_permutations=1` para
    minimizar custo da permutação — só queremos a `distance`, o p-value
    é descartado.
    """
    import torch
    from alibi_detect.cd import MMDDrift

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    detector = MMDDrift(
        x_ref=x_a,
        backend="pytorch",
        p_val=0.05,
        x_ref_preprocessed=True,
        preprocess_at_init=False,
        n_permutations=1,
        device=device,
    )
    data = detector.predict(x_b)["data"]
    return float(data["distance"])


def mmd2_consecutivo(
    embeddings_por_janela: list[np.ndarray],
    rotulos: list[str],
    *,
    seed: int = SEED,
    device: str | None = None,
) -> list[ResultadoSemantica]:
    """MMD² entre janelas adjacentes; `len(rotulos) − 1` resultados."""
    _validar_paridade(embeddings_por_janela, rotulos)
    resultados: list[ResultadoSemantica] = []
    for i in range(len(embeddings_por_janela) - 1):
        # Seed por par mantém determinismo entre execuções repetidas
        # (mesma convenção de run_b1.py para KTS/LSDD).
        valor = mmd2_estatistica(
            embeddings_por_janela[i],
            embeddings_por_janela[i + 1],
            seed=seed + i,
            device=device,
        )
        resultados.append(
            ResultadoSemantica(
                metrica="mmd2",
                janela_a=rotulos[i],
                janela_b=rotulos[i + 1],
                valor=valor,
            )
        )
    return resultados


NOMES_METRICAS_B2: tuple[str, ...] = ("cosine_centroid", "cumulative_cosine", "mmd2")


def aplicar_todas(
    embeddings_por_janela: list[np.ndarray],
    rotulos: list[str],
    *,
    seed: int = SEED,
    device: str | None = None,
) -> list[ResultadoSemantica]:
    """
    Aplica as três métricas em sequência. Cosseno consecutivo e
    cumulativo são independentes de seed; MMD² usa `seed` como base do
    par-a-par (vide `mmd2_consecutivo`).
    """
    resultados: list[ResultadoSemantica] = []
    resultados.extend(cosseno_centroide_consecutivo(embeddings_por_janela, rotulos))
    resultados.extend(cosseno_centroide_cumulativo(embeddings_por_janela, rotulos))
    resultados.extend(
        mmd2_consecutivo(embeddings_por_janela, rotulos, seed=seed, device=device)
    )
    return resultados


def _validar_paridade(
    embeddings_por_janela: list[np.ndarray], rotulos: list[str]
) -> None:
    if len(embeddings_por_janela) != len(rotulos):
        raise ValueError(
            f"len(embeddings_por_janela)={len(embeddings_por_janela)} != "
            f"len(rotulos)={len(rotulos)}."
        )
    if len(embeddings_por_janela) < 2:
        raise ValueError(
            "B2 exige pelo menos 2 janelas para gerar pares consecutivos."
        )
