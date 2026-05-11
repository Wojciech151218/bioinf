from __future__ import annotations

from ortools.sat.python import cp_model

from dna import Dna

from .matrix_pipeline import build_full_cost_matrices


def _expansion_for_dna(dna: Dna) -> list[tuple[int, int]]:
    expanded: list[tuple[int, int]] = []
    for ki, kmer in enumerate(dna.kmers):
        m = len(kmer.mapped_occurrences)
        for copy_ix in range(m):
            expanded.append((ki, copy_ix))
    return expanded


def _expanded_cost_and_gain(
    dna: Dna,
    expansion: list[tuple[int, int]],
    out_cost: list[list[int]],
) -> tuple[list[list[int]], list[list[int]]]:
    n = len(expansion)
    cost_e = [[0] * n for _ in range(n)]
    gain_e = [[0] * n for _ in range(n)]
    for u in range(n):
        ku, _ = expansion[u]
        for v in range(n):
            kv, _ = expansion[v]
            cost_e[u][v] = out_cost[ku][kv]
            if ku != kv:
                gain_e[u][v] = dna.length
    return cost_e, gain_e


def solve_selective_tsp(
    dna: Dna,
    out_cost: list[list[int]] | None = None,
    in_cost: list[list[int]] | None = None,
    *,
    log_progress: bool = False,
    linearization_level: int = 2,
    max_time_seconds: float | None = None,
    num_workers: int = 8,
) -> list[int]:

    if out_cost is None or in_cost is None:
        out_cost, in_cost = build_full_cost_matrices(dna)
    else:
        nkm = len(dna.kmers)
        if len(out_cost) != nkm or any(len(row) != nkm for row in out_cost):
            raise ValueError("out_cost must be nkmers × nkmers")
        for i in range(nkm):
            for j in range(nkm):
                if in_cost[i][j] != out_cost[j][i]:
                    raise ValueError("in_cost must be the transpose of out_cost")

    expansion = _expansion_for_dna(dna)
    n = len(expansion)
    if n == 0:
        return []

    cost_e, gain_e = _expanded_cost_and_gain(dna, expansion, out_cost)

    model = cp_model.CpModel()
    arc_lit: dict[tuple[int, int], cp_model.IntVar] = {}
    arcs: list[tuple[int, int, cp_model.IntVar]] = []

    obj_terms: list[cp_model.LinearExpr] = []

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            lit = model.new_bool_var(f"arc_{i}_{j}")
            arc_lit[i, j] = lit
            arcs.append((i, j, lit))
            coef = cost_e[i][j] - gain_e[i][j]
            obj_terms.append(coef * lit)

    if n == 1:
        loop = model.new_bool_var("loop_0")
        model.add(loop == 1)
        arcs = [(0, 0, loop)]
        obj_terms = []
    else:
        model.add_circuit(arcs)

    if obj_terms:
        model.minimize(cp_model.LinearExpr.sum(obj_terms))

    solver = cp_model.CpSolver()
    solver.parameters.log_search_progress = log_progress
    solver.parameters.linearization_level = linearization_level
    solver.parameters.num_search_workers = num_workers
    if max_time_seconds is not None:
        solver.parameters.max_time_in_seconds = max_time_seconds

    status = solver.solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise RuntimeError(f"CP-SAT could not find a tour: {solver.StatusName(status)}")

    if n == 1:
        return [expansion[0][0]]

    start = 0
    order_exp: list[int] = [start]
    current = start
    while True:
        nxt = None
        for j in range(n):
            if j == current:
                continue
            lit = arc_lit[current, j]
            if solver.boolean_value(lit):
                nxt = j
                break
        if nxt is None:
            raise RuntimeError("Broken tour: no outgoing arc")
        current = nxt
        if current == start:
            break
        order_exp.append(current)

    if len(order_exp) != n:
        raise RuntimeError("Tour does not visit every expanded node once")

    return [expansion[e][0] for e in order_exp]


def solve_from_dna(
    dna: Dna,
    *,
    log_progress: bool = False,
    linearization_level: int = 2,
    max_time_seconds: float | None = None,
    num_workers: int = 8,
) -> list[int]:
    out_c, in_c = build_full_cost_matrices(dna)
    return solve_selective_tsp(
        dna,
        out_c,
        in_c,
        log_progress=log_progress,
        linearization_level=linearization_level,
        max_time_seconds=max_time_seconds,
        num_workers=num_workers,
    )
