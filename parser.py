from pathlib import Path
from typing import IO
from xml.etree import ElementTree

from dna import Dna, Kmer


def _xml_root(source: str | Path | IO[str]) -> ElementTree.Element:
    if isinstance(source, Path):
        return ElementTree.parse(source).getroot()
    if isinstance(source, str):
        if source.lstrip().startswith("<"):
            return ElementTree.fromstring(source)
        return ElementTree.parse(source).getroot()
    return ElementTree.parse(source).getroot()


def parse_dna_xml(
    source: str | Path | IO[str],
    seed: int | None = None,
    *,
    max_cells: int | None = None,
) -> Dna:
    root = _xml_root(source)
    probe = root.find("probe")

    if probe is None:
        raise ValueError("XML file does not contain a <probe> element")

    pattern = probe.attrib.get("pattern", "")
    kmer_length = len(pattern)
    cells = probe.findall("cell")
    if max_cells is not None and max_cells > 0:
        cells = cells[:max_cells]

    kmers = [
        Kmer(
            sequence=(cell.text or "").strip(),
            intensity=int(cell.attrib["intensity"]),
        )
        for cell in cells
    ]

    if not kmer_length and kmers:
        kmer_length = len(kmers[0].sequence)

    dna = Dna(
        key=root.attrib["key"],
        length=int(root.attrib["length"]),
        start=root.attrib["start"],
        kmer_length=kmer_length,
        kmers=kmers,
    )
    dna.map_occurrences(seed=seed)

    return dna


if __name__ == "__main__":
    dna = parse_dna_xml("data.xml")
    print(dna)
