from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from math import exp
from random import Random
from time import perf_counter
from typing import Optional

from pathlib import Path
import sys

_THIS_FILE = Path(__file__).resolve()

if _THIS_FILE.parent.name == "heuristic":
    PROJECT_ROOT = _THIS_FILE.parent.parent
else:
    PROJECT_ROOT = _THIS_FILE.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from parser import parse_dna_xml

from dna import Dna, occurrence_range_for_intensity


DNA_ALPHABET = "ACGT"


@dataclass
class HeuristicResult:
    sequence: str
    cost: float
    iterations: int
    known_kmers_used: int
    unknown_kmers_count: int


class SimulatedAnnealingSolver:
    def __init__(
        self,
        dna: Dna,
        seed: Optional[int] = None,
        iterations: int = 60000,
        start_temperature: float = 80.0,
        end_temperature: float = 0.001,
        cooling: float = 0.9995,
        alpha: float = 10.0,
        beta: float = 1.0,
        time_limit: Optional[float] = None,
    ):
        self.dna = dna
        self.rng = Random(seed)

        self.iterations = iterations
        self.start_temperature = start_temperature
        self.end_temperature = end_temperature
        self.cooling = cooling

        self.time_limit = time_limit

        # alpha - kara za zla liczbe znanych l-merow
        # beta  - kara za l-mery, ktorych nie ma w pliku XML
        self.alpha = alpha
        self.beta = beta

        self.n = dna.length
        self.l = dna.kmer_length
        self.start = dna.start
        self.start_len = len(dna.start)

        self.spectrum = {}
        for kmer in dna.kmers:
            self.spectrum[kmer.sequence] = kmer.intensity

        self.spectrum_set = set(self.spectrum.keys())
        self.kmer_list = list(self.spectrum.keys())

        self.prefix_map = self._build_prefix_map()

    def _build_prefix_map(self) -> dict[str, list[str]]:
        result: dict[str, list[str]] = {}

        for kmer in self.kmer_list:
            if len(kmer) != self.l:
                continue

            prefix = kmer[:-1]

            if prefix not in result:
                result[prefix] = []

            result[prefix].append(kmer)

        return result

    def _random_letter(self) -> str:
        return self.rng.choice(DNA_ALPHABET)

    def _random_letter_different_than(self, old: str) -> str:
        new = self._random_letter()

        while new == old:
            new = self._random_letter()

        return new

    def random_dna(self, size: int) -> str:
        tekst = ""

        for _ in range(size):
            tekst += self._random_letter()

        return tekst

    def create_start_solution(self) -> str:
        seq = self.start

        while len(seq) < self.n:
            if len(seq) >= self.l - 1:
                suffix = seq[-(self.l - 1):]
                possible = self.prefix_map.get(suffix, [])

                if possible:
                    chosen = self.rng.choice(possible)
                    seq += chosen[-1]
                else:
                    seq += self._random_letter()
            else:
                seq += self._random_letter()

        return seq[:self.n]

    def count_lmers(self, sequence: str) -> Counter[str]:
        counts: Counter[str] = Counter()

        for i in range(len(sequence) - self.l + 1):
            one = sequence[i:i + self.l]
            counts[one] += 1

        return counts

    def intensity_range(self, intensity: int) -> tuple[int, int]:
        if intensity >= 9:
            return 9, self.n - self.l + 1

        return occurrence_range_for_intensity(intensity)

    def interval_penalty(self, value: int, low: int, high: int) -> int:
        if value < low:
            return (low - value) ** 2

        if value > high:
            return (value - high) ** 2

        return 0

    def evaluate(self, sequence: str) -> float:
        counts = self.count_lmers(sequence)
        cost = 0.0

        # Kara za znane l-mery, ktore wystepuja za malo albo za duzo.
        for kmer, intensity in self.spectrum.items():
            low, high = self.intensity_range(intensity)
            ile_razy = counts.get(kmer, 0)

            cost += self.alpha * self.interval_penalty(ile_razy, low, high)

        for kmer, ile_razy in counts.items():
            if kmer not in self.spectrum_set:
                cost += self.beta * ile_razy

        return cost

    def mutation_one_letter(self, sequence: str) -> str:
        if self.start_len >= len(sequence):
            return sequence

        seq = list(sequence)
        pos = self.rng.randint(self.start_len, len(seq) - 1)

        seq[pos] = self._random_letter_different_than(seq[pos])

        return "".join(seq)

    def mutation_block(self, sequence: str) -> str:
        if self.start_len >= len(sequence):
            return sequence

        seq = list(sequence)

        max_block = min(8, len(seq) - self.start_len)
        block_size = self.rng.randint(2, max_block)
        pos = self.rng.randint(self.start_len, len(seq) - block_size)

        for i in range(pos, pos + block_size):
            seq[i] = self._random_letter_different_than(seq[i])

        return "".join(seq)

    def put_existing_kmer(self, sequence: str) -> str:
        if not self.kmer_list:
            return sequence

        if self.n < self.l:
            return sequence

        seq = list(sequence)
        kmer = self.rng.choice(self.kmer_list)

        min_pos = self.start_len
        max_pos = self.n - self.l

        if min_pos > max_pos:
            return sequence

        pos = self.rng.randint(min_pos, max_pos)

        for i, znak in enumerate(kmer):
            seq[pos + i] = znak

        return "".join(seq)

    def rebuild_small_fragment(self, sequence: str) -> str:
        if self.start_len >= len(sequence):
            return sequence

        seq = list(sequence)

        max_size = min(25, len(seq) - self.start_len)
        size = self.rng.randint(8, max_size)
        pos = self.rng.randint(self.start_len, len(seq) - size)

        for i in range(pos, pos + size):
            seq[i] = self._random_letter()

        return "".join(seq)

    def make_neighbour(self, sequence: str) -> str:
        x = self.rng.random()

        if x < 0.55:
            return self.mutation_one_letter(sequence)

        if x < 0.75:
            return self.mutation_block(sequence)

        if x < 0.95:
            return self.put_existing_kmer(sequence)

        return self.rebuild_small_fragment(sequence)

    def accept_worse_solution(self, old_cost: float, new_cost: float, temperature: float) -> bool:
        difference = new_cost - old_cost

        if difference <= 0:
            return True

        probability = exp(-difference / temperature)

        return self.rng.random() < probability

    def statistics(self, sequence: str) -> tuple[int, int]:
        counts = self.count_lmers(sequence)

        known = 0
        unknown = 0

        for kmer, value in counts.items():
            if kmer in self.spectrum_set:
                known += value
            else:
                unknown += value

        return known, unknown

    def solve(self) -> HeuristicResult:
        deadline = None
        if self.time_limit is not None:
            deadline = perf_counter() + self.time_limit

        best: Optional[str] = None
        best_cost = float("inf")
        total_iterations = 0
        out_of_time = False

        while True:
            current = self.create_start_solution()
            current_cost = self.evaluate(current)

            if current_cost < best_cost:
                best = current
                best_cost = current_cost

            temperature = self.start_temperature

            for i in range(self.iterations):
                neighbour = self.make_neighbour(current)
                neighbour_cost = self.evaluate(neighbour)

                if self.accept_worse_solution(current_cost, neighbour_cost, temperature):
                    current = neighbour
                    current_cost = neighbour_cost

                if current_cost < best_cost:
                    best = current
                    best_cost = current_cost

                temperature *= self.cooling

                if temperature < self.end_temperature:
                    temperature = self.end_temperature

                total_iterations += 1

                # Czas sprawdzamy co 1024 iteracje, by nie wolac zegara zbyt czesto.
                if deadline is not None and (i & 1023) == 0 and perf_counter() >= deadline:
                    out_of_time = True
                    break

            if deadline is None or out_of_time or perf_counter() >= deadline:
                break

        known, unknown = self.statistics(best)

        return HeuristicResult(
            sequence=best,
            cost=best_cost,
            iterations=total_iterations,
            known_kmers_used=known,
            unknown_kmers_count=unknown,
        )


def simulated_annealing(
    dna: Dna,
    seed: Optional[int] = None,
    iterations: int = 60000,
    start_temperature: float = 80.0,
    cooling: float = 0.9995,
    alpha: float = 10.0,
    beta: float = 1.0,
    time_limit: Optional[float] = None,
) -> HeuristicResult:
    solver = SimulatedAnnealingSolver(
        dna=dna,
        seed=seed,
        iterations=iterations,
        start_temperature=start_temperature,
        cooling=cooling,
        alpha=alpha,
        beta=beta,
        time_limit=time_limit,
    )

    return solver.solve()


def basic_check(dna, result):
    seq = result.sequence

    print()
    print("=== BASIC CHECK ===")

    print("Dlugosc OK:", len(seq) == dna.length)
    print("Start OK:", seq.startswith(dna.start))
    print("Alfabet OK:", all(x in "ACGT" for x in seq))

    expected_lmers_count = dna.length - dna.kmer_length + 1
    real_lmers_count = len(seq) - dna.kmer_length + 1

    print("Liczba l-merow OK:", expected_lmers_count == real_lmers_count)
    print("Oczekiwana liczba l-merow:", expected_lmers_count)
    print("Rzeczywista liczba l-merow:", real_lmers_count)


if __name__ == "__main__":

    dna = parse_dna_xml("data.xml", seed=123)

    result = simulated_annealing(
        dna,
        seed=123,
        iterations=200000,
        start_temperature=80.0,
        cooling=0.9995,
        alpha=60.0,
        beta=1.0,
        time_limit=99.0,
        # alpha - kara za zla liczbe znanych l-merow
        # beta  - kara za l-mery, ktorych nie ma w pliku XML
    )

    print("Najlepszy koszt:", result.cost)
    print("Liczba iteracji:", result.iterations)
    print("Znane l-mery w wyniku:", result.known_kmers_used)
    print("Nieznane l-mery w wyniku:", result.unknown_kmers_count)
    print("Dlugosc sekwencji:", len(result.sequence))
    print("Sekwencja:")
    print(result.sequence)

    basic_check(dna, result)