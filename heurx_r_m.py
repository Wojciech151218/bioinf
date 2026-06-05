from __future__ import annotations

import sys
from io import StringIO

from parser import parse_dna_xml


def main() -> None:
    from heuristic.heuristic import simulated_annealing

    dna = parse_dna_xml(StringIO(sys.stdin.read()))
    result = simulated_annealing(dna)
    print(result.sequence)


if __name__ == "__main__":
    main()
