"""Regenerate the checked-in ReliabilityOS sample CSV.

Run with ``python -m data.generate_sample`` from the repository root.
"""

from .sample import SAMPLE_DATA_PATH, generate_sample_dataset


def main() -> None:
    frame = generate_sample_dataset()
    frame.to_csv(SAMPLE_DATA_PATH, index=False)
    print(f"Wrote {len(frame)} deterministic rows to {SAMPLE_DATA_PATH}")


if __name__ == "__main__":
    main()
