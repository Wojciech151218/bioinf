from dataclasses import dataclass, field
from random import Random


INTENSITY_TO_OCCURRENCE_RANGE: dict[int, tuple[int, int]] = {
    0: (0, 0),
    1: (1, 3),
    2: (3, 5),
    3: (5, 7),
    4: (6, 8),
    5: (7, 8),
    6: (7, 9),
    7: (8, 9),
    8: (8, 9),
}


@dataclass
class Kmer:
    sequence: str
    intensity: int
    mapped_occurrences: list[int] = field(default_factory=list)




@dataclass
class Dna:
    key: str
    length: int
    start: str
    kmer_length: int
    kmers: list[Kmer] = field(default_factory=list)

    def map_occurrences(self, seed: int | None = None) -> None:
        rng = Random(seed)
        for kmer in self.kmers:
            kmer.mapped_occurrences = map_occurrences(
                kmer.intensity,
                self.length,
                self.kmer_length,
                rng,
            )


def map_occurrences(
    intensity: int,
    dna_length: int,
    kmer_length: int,
    rng: Random | None = None,
) -> list[int]:
    rng = rng or Random()
    occurrence_count = random_occurrence_count(intensity, rng)
    possible_starts = max(dna_length - kmer_length + 1, 0)

    if occurrence_count == 0 or possible_starts == 0:
        return []

    if occurrence_count <= possible_starts:
        return sorted(rng.sample(range(possible_starts), occurrence_count))

    return sorted(rng.randrange(possible_starts) for _ in range(occurrence_count))


def random_occurrence_count(intensity: int, rng: Random | None = None) -> int:
    rng = rng or Random()
    low, high = occurrence_range_for_intensity(intensity)
    return rng.randint(low, high)


def occurrence_range_for_intensity(intensity: int) -> tuple[int, int]:
    if intensity >= 9:
        return (9, 9)

    try:
        return INTENSITY_TO_OCCURRENCE_RANGE[intensity]
    except KeyError as error:
        raise ValueError(f"Unsupported intensity: {intensity}") from error
