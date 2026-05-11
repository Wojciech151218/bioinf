from __future__ import annotations

import argparse
from pathlib import Path

from parser import parse_dna_xml


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="DNA k-mer assembly pipeline")
    p.add_argument(
        "algorithm",
        choices=["optimal", "heuristic"],
        help="Assembly algorithm: optimal (CP-SAT TSP) or heuristic (simulated annealing)",
    )
    p.add_argument(
        "max_rows",
        type=int,
        nargs="?",
        default=None,
        help="Max <cell> rows from XML (default: entire file). 0 or negative also means all",
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
    p.add_argument(
        "--iterations",
        type=int,
        default=60_000,
        help="Simulated annealing iterations (heuristic only, default: 60000)",
    )
    p.add_argument(
        "--check",
        action="store_true",
        help="Run basic_check() on heuristic result (heuristic only)",
    )
    p.add_argument(
        "--alpha",
        type=float,
        default=10.0,
        help="Penalty weight for known k-mer count bands (heuristic only, default: 10)",
    )
    p.add_argument(
        "--beta",
        type=float,
        default=1.0,
        help="Penalty weight for k-mers not in spectrum (heuristic only, default: 1)",
    )
    return p


def _max_cells(max_rows: int | None) -> int | None:
    if max_rows is None or max_rows <= 0:
        return None
    return max_rows


def _run_optimal(args: argparse.Namespace) -> None:
    from optimal.solver import solve_from_dna
    from optimal.solution_transformer import transform

    dna = parse_dna_xml(args.xml, seed=args.seed, max_cells=_max_cells(args.max_rows))
    path = solve_from_dna(dna, max_time_seconds=args.time_limit)
    ordered, sequence = transform(dna, path)
    print("K-mer indices along tour:", path)
    print("Merged sequence length:", len(sequence))
    print(sequence)


def _run_heuristic(args: argparse.Namespace) -> None:
    from heuristic.heuristic import basic_check, simulated_annealing

    dna = parse_dna_xml(args.xml, seed=args.seed, max_cells=_max_cells(args.max_rows))
    result = simulated_annealing(
        dna,
        seed=args.seed,
        iterations=args.iterations,
        alpha=args.alpha,
        beta=args.beta,
    )
    print("Best cost:", result.cost)
    print("Iterations:", result.iterations)
    print("Known k-mer windows in result:", result.known_kmers_used)
    print("Unknown k-mer windows in result:", result.unknown_kmers_count)
    print("Sequence length:", len(result.sequence))
    print("Sequence:")
    print(result.sequence)
    if args.check:
        basic_check(dna, result)


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
