"""
Bloco B1 (testes estatísticos) da detecção de drift (ADRs 010 e 011).

Quatro testes via `alibi-detect`:

- **KS** (Kolmogorov-Smirnov): univariado por dimensão, agregado via
  método de Fisher para um único `p_value` multivariado.
- **CVM** (Cramér-von Mises): idem, univariado + Fisher.
- **KTS** (Kernel Two-Sample Test, `MMDDrift` com kernel RBF gaussiano):
  nativamente multivariado, p_value via permutation test.
- **LSDD** (Least-Squares Density Difference): nativamente multivariado,
  p_value via permutation test.

Retorno uniforme em `Resultado(p_value, estatistica)`:

- KS/CVM: `estatistica` é o χ² de Fisher (soma de log p-values).
- KTS: `estatistica` é o MMD² observado.
- LSDD: `estatistica` é o valor LSDD observado.

Determinismo: KTS e LSDD usam permutation test — passar `seed` explícita.
KS e CVM são determinísticos (scipy puro).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import chi2

from src.config import SEED


_EPS_LOG: float = 1e-300


@dataclass(frozen=True)
class Resultado:
    p_value: float
    estatistica: float


def fisher_combinar(p_values: np.ndarray) -> tuple[float, float]:
    """
    Combina p-values univariados em um único via método de Fisher.

    χ² = −2 · Σ log(pᵢ),  com df = 2k onde k é o número de p-values.
    Retorna `(p_value_combinado, chi2_stat)`.
    """
    if len(p_values) == 0:
        raise ValueError("Não há p-values para combinar via Fisher.")
    # Upcast para float64 antes do clip: 1e-300 underflow para 0 em float32.
    p_safe = np.clip(np.asarray(p_values, dtype=np.float64), _EPS_LOG, 1.0)
    chi2_stat = float(-2.0 * np.log(p_safe).sum())
    df = 2 * len(p_values)
    return float(chi2.sf(chi2_stat, df=df)), chi2_stat


def aplicar_ks(x_a: np.ndarray, x_b: np.ndarray) -> Resultado:
    """Kolmogorov-Smirnov por dimensão + agregação Fisher."""
    from alibi_detect.cd import KSDrift

    detector = KSDrift(
        x_ref=x_a,
        p_val=0.05,
        x_ref_preprocessed=True,
        preprocess_at_init=False,
    )
    data = detector.predict(x_b)["data"]
    p_combinado, chi2_stat = fisher_combinar(np.asarray(data["p_val"]))
    return Resultado(p_value=p_combinado, estatistica=chi2_stat)


def aplicar_cvm(x_a: np.ndarray, x_b: np.ndarray) -> Resultado:
    """Cramér-von Mises por dimensão + agregação Fisher."""
    from alibi_detect.cd import CVMDrift

    detector = CVMDrift(
        x_ref=x_a,
        p_val=0.05,
        x_ref_preprocessed=True,
        preprocess_at_init=False,
    )
    data = detector.predict(x_b)["data"]
    p_combinado, chi2_stat = fisher_combinar(np.asarray(data["p_val"]))
    return Resultado(p_value=p_combinado, estatistica=chi2_stat)


def aplicar_kts(
    x_a: np.ndarray,
    x_b: np.ndarray,
    *,
    n_permutations: int = 100,
    seed: int = SEED,
    device: str | None = None,
) -> Resultado:
    """
    KTS via MMDDrift (kernel RBF gaussiano com bandwidth via heurística
    da mediana — default do alibi-detect). Permutation test com
    `n_permutations` permutações.
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
        n_permutations=n_permutations,
        device=device,
    )
    data = detector.predict(x_b)["data"]
    return Resultado(
        p_value=float(data["p_val"]),
        estatistica=float(data["distance"]),
    )


def aplicar_lsdd(
    x_a: np.ndarray,
    x_b: np.ndarray,
    *,
    n_permutations: int = 100,
    seed: int = SEED,
    device: str | None = None,
) -> Resultado:
    """
    LSDD com `n_permutations` permutações para o p-value.
    """
    import torch
    from alibi_detect.cd import LSDDDrift

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    detector = LSDDDrift(
        x_ref=x_a,
        backend="pytorch",
        p_val=0.05,
        x_ref_preprocessed=True,
        preprocess_at_init=False,
        n_permutations=n_permutations,
        device=device,
    )
    data = detector.predict(x_b)["data"]
    return Resultado(
        p_value=float(data["p_val"]),
        estatistica=float(data["distance"]),
    )


TESTES_UNIVARIADOS: tuple[str, ...] = ("KS", "CVM")
TESTES_MULTIVARIADOS: tuple[str, ...] = ("KTS", "LSDD")


def aplicar_todos(
    x_a: np.ndarray,
    x_b: np.ndarray,
    *,
    n_permutations: int = 100,
    seed: int = SEED,
    device: str | None = None,
) -> dict[str, Resultado]:
    """
    Aplica os quatro testes ao par (x_a, x_b) e devolve um dicionário
    `{nome_teste: Resultado}`. Ordem: KS, CVM, KTS, LSDD.
    """
    return {
        "KS": aplicar_ks(x_a, x_b),
        "CVM": aplicar_cvm(x_a, x_b),
        "KTS": aplicar_kts(
            x_a, x_b, n_permutations=n_permutations, seed=seed, device=device
        ),
        "LSDD": aplicar_lsdd(
            x_a, x_b, n_permutations=n_permutations, seed=seed, device=device
        ),
    }
