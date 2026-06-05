from __future__ import annotations

import sys
from io import StringIO

from parser import parse_dna_xml


def main() -> None:
    from optimal.solver import solve_from_dna
    from optimal.solution_transformer import transform

    dna = parse_dna_xml(StringIO(sys.stdin.read()))
    path = solve_from_dna(dna)
    _, sequence = transform(dna, path)
    print(sequence)


if __name__ == "__main__":
    main()
