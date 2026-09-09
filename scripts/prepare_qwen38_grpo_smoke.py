"""Create a tiny deterministic text dataset for GRPO infrastructure validation."""

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    """A synthetic text-format reward, not a benchmark accuracy metric."""
    text = solution_str.strip()
    exact = float(text == str(ground_truth))
    if not text:
        return {"score": 0.0, "exact_match": exact, "format_score": 0.0}
    printable = sum(c.isascii() and c.isprintable() for c in text) / len(text)
    diversity = len(set(text)) / len(text)
    format_score = 0.5 * printable + 0.5 * diversity
    return {"score": exact + format_score, "exact_match": exact, "format_score": format_score}


def main():
    output = Path(__file__).resolve().parents[1] / "artifacts/qwen38-grpo/data"
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, (prompt, answer) in enumerate(
        (
            ("Calculate 1 + 1. Reply with the answer only.", "2"),
            ("Calculate 2 + 3. Reply with the answer only.", "5"),
            ("Calculate 4 + 1. Reply with the answer only.", "5"),
            ("Calculate 2 + 2. Reply with the answer only.", "4"),
        )
    ):
        rows.append(
            {
                "data_source": "qwen38_grpo_smoke",
                "prompt": [{"role": "user", "content": prompt}],
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": answer},
                "extra_info": {"index": index, "split": "train"},
            }
        )
    pq.write_table(pa.Table.from_pylist(rows), output / "train.parquet")
    pq.write_table(pa.Table.from_pylist(rows[:2]), output / "val.parquet")
    (output / "reward_description.json").write_text(
        json.dumps(
            {
                "purpose": "GRPO execution smoke only",
                "reward": "exact_answer + (printable_ascii_fraction + character_diversity)/2",
            },
            indent=2,
        )
        + "\n"
    )
    print(output)


if __name__ == "__main__":
    main()
