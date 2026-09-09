"""Read-only structural and metric checks for the frozen-PLE GRPO smoke run."""

import argparse
import json
import math
import re
import zipfile
from collections import Counter
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    args = parser.parse_args()
    log = args.log.read_text()
    verified = Counter(re.findall(r"Frozen PLE verified after optimizer step on rank (\d+)", log))
    # Rank 0 did not emit post-step logger messages in the successful run.
    # Require both messages for ranks 1-7; report rank 0 as missing, not verified by logs.
    assert all(verified[str(rank)] == 2 for rank in range(1, 8)), verified
    assert verified["0"] in (0, 2), verified
    assert "Frozen PLE invariant violated" not in log
    initialized = set(re.findall(r"Frozen PLE: 128 tables, 11.921 GiB local on rank (\d+)", log))
    assert initialized == {str(rank) for rank in range(8)}, initialized
    assert "grad_norm is not finite" not in log
    metrics_path = args.artifact / "logs/qwen38_veomni/3layer_frozen_ple_grpo_8npu.jsonl"
    records = [json.loads(line) for line in metrics_path.read_text().splitlines()]
    assert [record["step"] for record in records] == [1, 2]
    metrics = []
    for record in records:
        data = record["data"]
        norm = data["actor/grad_norm"]
        assert math.isfinite(norm) and norm > 0
        assert data["timing_s/update_weights"] > 0
        assert data["critic/advantages/max"] > data["critic/advantages/min"]
        metrics.append({
            "step": record["step"],
            "gradient_norm": norm,
            "reward_mean": data["critic/rewards/mean"],
            "actor_peak_allocated_gib": data["actor/perf/max_memory_allocated_gb"],
            "weight_sync_seconds": data["timing_s/update_weights"],
        })
    checkpoint = args.artifact / "checkpoints/global_step_2/actor"
    assert (args.artifact / "checkpoints/latest_checkpointed_iteration.txt").read_text().strip() == "2"
    table_key = re.compile(rb"ple\.ple_embedding\.ngram_embedding\.shard_\d+\.weight")
    counts = Counter()
    total_bytes = 0
    for kind in ("model", "optim", "extra_state"):
        for rank in range(8):
            path = checkpoint / f"{kind}_world_size_8_rank_{rank}.pt"
            with zipfile.ZipFile(path) as archive:
                names = archive.namelist()
                metadata = [name for name in names if name.endswith("/data.pkl")]
                assert len(metadata) == 1, path
                payload = archive.read(metadata[0])
                keys = set(table_key.findall(payload))
                if kind == "model":
                    assert len(keys) == 128, (path, len(keys))
                elif kind == "optim":
                    assert not keys, path
            counts[kind] += 1
            total_bytes += path.stat().st_size
    rollout_counts = {}
    for step in (1, 2):
        samples = [json.loads(line) for line in (args.artifact / f"rollouts/{step}.jsonl").read_text().splitlines()]
        assert len(samples) == 8, (step, len(samples))
        rollout_counts[step] = len(samples)
    print(json.dumps({
        "metrics": metrics,
        "freeze_checks_per_rank_in_log": dict(sorted(verified.items())),
        "missing_post_step_log_ranks": [rank for rank in range(8) if not verified[str(rank)]],
        "checkpoint_files": dict(counts),
        "checkpoint_total_gib": total_bytes / 2**30,
        "saved_ple_keys_per_model_rank": 128,
        "saved_ple_optimizer_keys": 0,
        "rollout_counts": rollout_counts,
        "verification_scope": "ZIP metadata and key checks; not full payload CRC, resume, or merged export",
    }, indent=2))


if __name__ == "__main__":
    main()
