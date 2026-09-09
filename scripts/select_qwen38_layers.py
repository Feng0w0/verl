"""Select and renumber text layers without modifying the source checkpoint."""

import argparse
import json
import re
import shutil
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file

LAYER = re.compile(r"^model\.language_model\.layers\.(\d+)\.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layers", type=int, nargs="+", required=True)
    args = parser.parse_args()
    config = json.loads((args.source / "config.json").read_text())
    tc = config["text_config"]
    assert args.layers == sorted(set(args.layers))
    assert all(0 <= i < tc["num_hidden_layers"] for i in args.layers)
    if args.output.exists():
        raise FileExistsError(args.output)
    mapping = {old: new for new, old in enumerate(args.layers)}
    index = json.loads((args.source / "model.safetensors.index.json").read_text())
    args.output.mkdir(parents=True)
    weight_map = {}
    total_size = 0
    for filename in sorted(set(index["weight_map"].values())):
        tensors = {}
        with safe_open(args.source / filename, framework="pt", device="cpu") as shard:
            for key in shard.keys():
                match = LAYER.match(key)
                if match:
                    old = int(match.group(1))
                    if old not in mapping:
                        continue
                    new_key = LAYER.sub(f"model.language_model.layers.{mapping[old]}.", key)
                else:
                    new_key = key
                tensors[new_key] = shard.get_tensor(key)
                weight_map[new_key] = filename
                total_size += tensors[new_key].numel() * tensors[new_key].element_size()
        if tensors:
            save_file(tensors, args.output / filename, metadata={"format": "pt"})
            print(filename, len(tensors), flush=True)
    tc["num_hidden_layers"] = len(args.layers)
    tc["layer_types"] = [tc["layer_types"][i] for i in args.layers]
    # Hugging Face Qwen4Exp ple_layer_ids use one-based decoder indices.
    tc["ple_layer_ids"] = [mapping[i - 1] + 1 for i in tc.get("ple_layer_ids", []) if i - 1 in mapping]
    (args.output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    for filename in (
        "chat_template.jinja",
        "generation_config.json",
        "preprocessor_config.json",
        "processor_config.json",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "video_preprocessor_config.json",
    ):
        if (args.source / filename).is_file():
            shutil.copy2(args.source / filename, args.output / filename)
    (args.output / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": total_size}, "weight_map": weight_map}, indent=2) + "\n"
    )
    (args.output / "reduction_manifest.json").write_text(
        json.dumps(
            {
                "source": str(args.source),
                "original_layer_indices": args.layers,
                "num_tensors": len(weight_map),
                "total_bytes": total_size,
                "ple_layer_ids": tc["ple_layer_ids"],
            },
            indent=2,
        )
        + "\n"
    )
    print(dict(tensors=len(weight_map), gib=total_size / 2**30, config=tc["layer_types"]), flush=True)


if __name__ == "__main__":
    main()
