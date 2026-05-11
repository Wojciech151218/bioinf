"""CLI: parse XML, run selected algorithm, print results."""

from __future__ import annotations

import argparse
from pathlib import Path

from parser import parse_dna_xml


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="DNA k-mer assembly pipeline")
    p.add_argument(
        "algorithm",
        choices=["optimal", "heuristic"],
        help="Assembly algorithm: optimal (CP-SAT TSP) or heuristic (not implemented)",
    )
    p.add_argument(
        "max_rows",
        type=int,
        help="Number of <cell> rows to read from XML; 0 or negative means all rows",
    )
    p.add_argument(
        "--xml",
        type=Path,
        default=Path("data.xml"),
        help="Path to DNA XML (default: data.xml)",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for mapping occurrence counts (default: OS random)",
    )
    p.add_argument(
        "--time-limit",
        type=float,
        default=None,
        help="CP-SAT time limit in seconds (optimal only)",
    )
    return p


def _run_optimal(args: argparse.Namespace) -> None:
    from optimal.solver import solve_from_dna
    from optimal.solution_transformer import transform

    max_cells = args.max_rows if args.max_rows > 0 else None
    dna = parse_dna_xml(args.xml, seed=args.seed, max_cells=max_cells)
    path = solve_from_dna(dna, max_time_seconds=args.time_limit)
    ordered, sequence = transform(dna, path)
    print("K-mer indices along tour:", path)
    print("Merged sequence length:", len(sequence))
    print(sequence)


def _run_heuristic(_args: argparse.Namespace) -> None:
    raise SystemExit(
        "Algorithm 'heuristic' is not implemented yet. Use 'optimal' or add a "
        "heuristic under heuristic/."
    )


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    if args.algorithm == "optimal":
        _run_optimal(args)
    elif args.algorithm == "heuristic":
        _run_heuristic(args)
    else:
        parser.error(f"Unknown algorithm: {args.algorithm}")


if __name__ == "__main__":
    main()
