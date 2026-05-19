from __future__ import annotations

from itertools import product

from ortools.sat.python import cp_model

from dna import Dna

from .matrix_pipeline import (
    MISSING_NUCLEOTIDE,
    build_full_cost_matrices,
    build_overlap_matrix,
    gap_kmer_index,
    is_gap_kmer_index,
    merge_sequences,
    node_sequence_length,
)


def _expansion_for_dna(dna: Dna) -> list[tuple[int, int]]:
    expanded: list[tuple[int, int]] = []
    for ki, kmer in enumerate(dna.kmers):
        m = len(kmer.mapped_occurrences)
        for copy_ix in range(m):
            expanded.append((ki, copy_ix))
    gap_ki = gap_kmer_index(dna)
    for copy_ix in range(dna.length):
        expanded.append((gap_ki, copy_ix))
    return expanded


def _is_gap_node(dna: Dna, expansion: list[tuple[int, int]], u: int) -> bool:
    ki, _ = expansion[u]
    return is_gap_kmer_index(dna, ki)


def _node_lengths(
    dna: Dna, expansion: list[tuple[int, int]]
) -> list[int]:
    return [node_sequence_length(dna, expansion[u][0]) for u in range(len(expansion))]


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


def _expanded_overlap(
    dna: Dna,
    expansion: list[tuple[int, int]],
    overlap: list[list[int]],
) -> list[list[int]]:
    n = len(expansion)
    overlap_e = [[0] * n for _ in range(n)]
    for u in range(n):
        ku, _ = expansion[u]
        for v in range(n):
            kv, _ = expansion[v]
            overlap_e[u][v] = overlap[ku][kv]
    return overlap_e


def _and_bool(
    model: cp_model.CpModel,
    name: str,
    literals: list[cp_model.IntVar],
) -> cp_model.IntVar:
    if len(literals) == 1:
        return literals[0]
    z = model.new_bool_var(name)
    for lit in literals:
        model.add(z <= lit)
    model.add(z >= cp_model.LinearExpr.sum(literals) - (len(literals) - 1))
    return z


def _seq_at_expanded(dna: Dna, expansion: list[tuple[int, int]], u: int) -> str:
    ki, _ = expansion[u]
    if is_gap_kmer_index(dna, ki):
        return MISSING_NUCLEOTIDE
    return dna.kmers[ki].sequence


def _prefix_kmers_needed(prefix_len: int, kmer_length: int) -> int:
    if prefix_len <= 0:
        return 0
    return (prefix_len + kmer_length - 1) // kmer_length


def _add_merged_length_constraint(
    model: cp_model.CpModel,
    arc_lit: dict[tuple[int, int], cp_model.IntVar],
    overlap_e: list[list[int]],
    first: list[cp_model.IntVar],
    node_len: list[int],
    gap_nodes: list[int],
    target_length: int,
) -> None:
    """Merged length on the open tour; gap nodes add 1, k-mers add k; gaps never overlap."""
    n = len(node_len)
    mandatory_len = sum(node_len[u] for u in range(n) if u not in gap_nodes)

    if n == len(gap_nodes) and n == 1:
        model.add(node_len[0] == target_length)
        return

    overlap_on_arcs = [
        overlap_e[i][j] * arc_lit[i, j]
        for i in range(n)
        for j in range(n)
        if i != j
    ]
    sum_all_overlaps = cp_model.LinearExpr.sum(overlap_on_arcs)
    closing_terms: list[cp_model.LinearExpr] = []
    for j in range(n):
        for u in range(n):
            if j == u:
                continue
            break_arc = _and_bool(
                model,
                f"break_{j}_{u}",
                [arc_lit[j, u], first[u]],
            )
            closing_terms.append(overlap_e[j][u] * break_arc)

    gap_active: list[cp_model.IntVar] = []
    for g in gap_nodes:
        if (g, g) not in arc_lit:
            continue
        active = model.new_bool_var(f"gap_active_{g}")
        model.add(active + arc_lit[g, g] == 1)
        gap_active.append(active)
    path_overlaps = sum_all_overlaps - cp_model.LinearExpr.sum(closing_terms)
    merged_len = mandatory_len + cp_model.LinearExpr.sum(gap_active) - path_overlaps
    model.add(merged_len == target_length)


def _add_start_prefix_constraint(
    model: cp_model.CpModel,
    dna: Dna,
    expansion: list[tuple[int, int]],
    arc_lit: dict[tuple[int, int], cp_model.IntVar],
    first: list[cp_model.IntVar],
    n: int,
) -> None:
    """Merged sequence must begin with dna.start (open tour from the first node)."""
    start = dna.start
    if not start:
        return

    k = dna.kmer_length
    prefix_len = len(start)
    t = _prefix_kmers_needed(prefix_len, k)
    if t == 0:
        return

    if t == 1:
        for u in range(n):
            if _is_gap_node(dna, expansion, u):
                model.add(first[u] == 0)
                continue
            seq = _seq_at_expanded(dna, expansion, u)
            if seq[:prefix_len] != start:
                model.add(first[u] == 0)
        return

    if t > 4:
        raise ValueError(
            f"dna.start length {prefix_len} needs {t} initial k-mers; "
            "supported up to 4 (increase cap or shorten start)"
        )

    good_path_literals: list[cp_model.IntVar] = []
    paths_at_start: dict[int, list[cp_model.IntVar]] = {u: [] for u in range(n)}
    for path in _iter_expanded_paths(n, t):
        seqs = [_seq_at_expanded(dna, expansion, u) for u in path]
        merged = merge_sequences(k, seqs)
        if merged[:prefix_len] != start:
            continue
        lits: list[cp_model.IntVar] = [first[path[0]]]
        for a, b in zip(path, path[1:]):
            lits.append(arc_lit[a, b])
        z = _and_bool(model, f"start_path_{'_'.join(map(str, path))}", lits)
        good_path_literals.append(z)
        paths_at_start[path[0]].append(z)

    if not good_path_literals:
        raise ValueError("No k-mer tour prefix matches dna.start")

    model.add(cp_model.LinearExpr.sum(good_path_literals) >= 1)
    for u in range(n):
        if paths_at_start[u]:
            model.add(first[u] <= cp_model.LinearExpr.sum(paths_at_start[u]))
        else:
            model.add(first[u] == 0)


def _iter_expanded_paths(n: int, length: int):
    if length <= 0:
        return
    if length == 1:
        for u in range(n):
            yield (u,)
        return
    for path in product(range(n), repeat=length):
        yield path


def solve_selective_tsp(
    dna: Dna,
    out_cost: list[list[int]] | None = None,
    in_cost: list[list[int]] | None = None,
    *,
    log_progress: bool = False,
    linearization_level: int = 2,
    max_time_seconds: float | None = None,
    num_workers: int = 8,
    enforce_target_length: bool = True,
    enforce_start_prefix: bool = True,
) -> list[int]:

    matrix_size = len(dna.kmers) + 1
    if out_cost is None or in_cost is None:
        out_cost, in_cost = build_full_cost_matrices(dna)
    else:
        if len(out_cost) != matrix_size or any(len(row) != matrix_size for row in out_cost):
            raise ValueError("out_cost must be (n_kmers + 1) × (n_kmers + 1) including gap row")
        for i in range(matrix_size):
            for j in range(matrix_size):
                if in_cost[i][j] != out_cost[j][i]:
                    raise ValueError("in_cost must be the transpose of out_cost")

    expansion = _expansion_for_dna(dna)
    n = len(expansion)
    if n == 0:
        return []

    gap_nodes = [u for u in range(n) if _is_gap_node(dna, expansion, u)]
    mandatory_nodes = [u for u in range(n) if u not in gap_nodes]
    node_len = _node_lengths(dna, expansion)

    cost_e, gain_e = _expanded_cost_and_gain(dna, expansion, out_cost)
    overlap_km = build_overlap_matrix(dna)
    overlap_e = _expanded_overlap(dna, expansion, overlap_km)

    model = cp_model.CpModel()
    arc_lit: dict[tuple[int, int], cp_model.IntVar] = {}
    arcs: list[tuple[int, int, cp_model.IntVar]] = []

    obj_terms: list[cp_model.LinearExpr] = []

    for i in range(n):
        for j in range(n):
            if i == j and i not in gap_nodes:
                continue
            lit = model.new_bool_var(f"arc_{i}_{j}")
            arc_lit[i, j] = lit
            arcs.append((i, j, lit))
            if i != j:
                coef = cost_e[i][j] - gain_e[i][j]
                obj_terms.append(coef * lit)

    if len(mandatory_nodes) == 0 and len(gap_nodes) == 1:
        loop = model.new_bool_var("loop_0")
        model.add(loop == 1)
        arcs = [(0, 0, loop)]
        arc_lit[0, 0] = loop
        obj_terms = []
    elif len(mandatory_nodes) == 1 and not gap_nodes:
        loop = model.new_bool_var("loop_0")
        model.add(loop == 1)
        arcs = [(mandatory_nodes[0], mandatory_nodes[0], loop)]
        arc_lit[mandatory_nodes[0], mandatory_nodes[0]] = loop
        obj_terms = []
    else:
        model.add_circuit(arcs)

    first: list[cp_model.IntVar] = []
    if n > 1 and (enforce_target_length or enforce_start_prefix):
        first = [model.new_bool_var(f"first_{u}") for u in range(n)]
        model.add(cp_model.LinearExpr.sum(first) == 1)
        for g in gap_nodes:
            model.add(first[g] == 0)

    if enforce_target_length:
        _add_merged_length_constraint(
            model,
            arc_lit,
            overlap_e,
            first,
            node_len,
            gap_nodes,
            dna.length,
        )

    if enforce_start_prefix:
        if len(mandatory_nodes) == 1 and not gap_nodes:
            seq = _seq_at_expanded(dna, expansion, mandatory_nodes[0])
            if dna.start and seq[: len(dna.start)] != dna.start:
                raise ValueError("Single k-mer does not match dna.start")
        elif n > 1:
            _add_start_prefix_constraint(
                model, dna, expansion, arc_lit, first, n
            )

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

    if len(mandatory_nodes) <= 1 and not gap_nodes:
        return [expansion[mandatory_nodes[0]][0]] if mandatory_nodes else []

    tour_start = mandatory_nodes[0] if mandatory_nodes else 0
    if first:
        for u in range(n):
            if solver.boolean_value(first[u]):
                tour_start = u
                break

    start = tour_start
    order_exp: list[int] = [start]
    current = start
    while True:
        nxt = None
        for j in range(n):
            if j == current:
                continue
            lit = arc_lit.get((current, j))
            if lit is not None and solver.boolean_value(lit):
                nxt = j
                break
        if nxt is None:
            raise RuntimeError("Broken tour: no outgoing arc")
        current = nxt
        if current == start:
            break
        order_exp.append(current)

    active_gap_count = sum(
        1
        for g in gap_nodes
        if (g, g) in arc_lit and not solver.boolean_value(arc_lit[g, g])
    )
    if len(order_exp) != len(mandatory_nodes) + active_gap_count:
        raise RuntimeError("Tour does not visit every active node once")

    return [expansion[e][0] for e in order_exp]


def solve_from_dna(
    dna: Dna,
    *,
    log_progress: bool = False,
    linearization_level: int = 2,
    max_time_seconds: float | None = None,
    num_workers: int = 8,
    enforce_target_length: bool = True,
    enforce_start_prefix: bool = True,
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
        enforce_target_length=enforce_target_length,
        enforce_start_prefix=enforce_start_prefix,
    )
