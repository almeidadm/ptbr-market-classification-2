"""
Executa um cenário de experimento (ADR 009).

Uso:
    # Primário (busca HP por fold):
    python scripts/experimentar.py --familia logreg --recorte opcao7 \\
        --protocolo rolling --regime multi

    # Ablation (reusa HPs do fold 0 do primário):
    python scripts/experimentar.py --familia logreg --recorte opcao4 \\
        --protocolo rolling --regime multi \\
        --hp-de artifacts/experimentos/20260424-1234-logreg-opcao7-rolling-multi

    # Smoke validation (subset estratificado por mês):
    python scripts/experimentar.py --familia logreg --recorte opcao7 \\
        --protocolo rolling --regime multi --smoke 1000
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

RAIZ_PROJETO = Path(__file__).resolve().parent.parent
if str(RAIZ_PROJETO) not in sys.path:
    sys.path.insert(0, str(RAIZ_PROJETO))

from src.config import FAMILIAS, PROTOCOLOS, RECORTES, REGIMES
from src.experimento.runner import executar_cenario


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--familia", required=True, choices=list(FAMILIAS))
    parser.add_argument("--recorte", required=True, choices=list(RECORTES))
    parser.add_argument("--protocolo", required=True, choices=list(PROTOCOLOS))
    parser.add_argument("--regime", required=True, choices=list(REGIMES))
    parser.add_argument(
        "--hp-de",
        type=Path,
        default=None,
        help="Diretório do experimento primário do qual herdar `best_hp.json` (regra ii).",
    )
    parser.add_argument(
        "--smoke",
        type=int,
        default=None,
        metavar="N",
        help="Executa em subset estratificado por mês com `N` artigos.",
    )
    args = parser.parse_args(argv)

    caminho = executar_cenario(
        familia=args.familia,
        recorte=args.recorte,
        protocolo=args.protocolo,
        regime=args.regime,
        hp_de=args.hp_de,
        subset_n=args.smoke,
    )
    print(f"Experimento concluído: {caminho}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
