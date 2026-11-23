"""Public ATC speech corpora, loaded from the Hugging Face mirrors by Juan Zuluaga-Gomez.

All three share the schema: id, audio, text, segment_start_time, segment_end_time, duration.
References are lowercase spoken forms.
"""

from __future__ import annotations

from datasets import Audio, Dataset, load_dataset

SAMPLING_RATE = 16_000

CORPORA = {
    # name: (hub repo, available splits, licence note)
    "atco2_1h": ("Jzuluaga/atco2_corpus_1h", ("test",), "ATCO2 free 1h test set"),
    "uwb_atcc": ("Jzuluaga/uwb_atcc", ("train", "test"), "CC BY-NC-SA 4.0, non-commercial"),
    "atcosim": ("Jzuluaga/atcosim_corpus", ("train", "test"), "simulated, controllers only, headset mics"),
}

# Models whose training data likely includes these corpora, so their scores there are
# in-distribution at best and contaminated at worst. Whisper-ATC's model cards don't list
# the training data; the paper trains on ATCO2 and ATCOSIM.
KNOWN_OVERLAP = {
    "jlvdoorn/whisper-large-v3-atco2-asr": {"atco2_1h"},
    "jlvdoorn/whisper-large-v3-atco2-asr-atcosim": {"atco2_1h", "atcosim"},
}


def parse_spec(spec: str) -> tuple[str, str]:
    """'uwb_atcc:test' -> ('uwb_atcc', 'test')."""
    name, _, split = spec.partition(":")
    if name not in CORPORA:
        raise ValueError(f"unknown corpus {name!r}; choose from {sorted(CORPORA)}")
    split = split or CORPORA[name][1][-1]
    if split not in CORPORA[name][1]:
        raise ValueError(f"{name} has splits {CORPORA[name][1]}, not {split!r}")
    return name, split


def load(spec: str, max_samples: int | None = None, seed: int = 0) -> Dataset:
    """Load a corpus split with audio decoded at 16 kHz.

    `max_samples` takes a fixed random subset (same seed -> same subset), so quick
    evaluations are comparable across models.
    """
    name, split = parse_spec(spec)
    ds = load_dataset(CORPORA[name][0], split=split)
    if max_samples is not None and max_samples < len(ds):
        ds = ds.shuffle(seed=seed).select(range(max_samples))
    return ds.cast_column("audio", Audio(sampling_rate=SAMPLING_RATE))
