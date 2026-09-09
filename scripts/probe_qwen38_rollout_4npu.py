"""Standalone vLLM-Ascend rollout probe, independent of Ray lifecycle."""

import os

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
from vllm import LLM, SamplingParams


def main():
    llm = LLM(
        model="/mnt/weight/Qwen3.8-Flash-Next-3layer-grpo",
        trust_remote_code=True,
        tensor_parallel_size=4,
        dtype="bfloat16",
        max_model_len=2048,
        max_num_seqs=8,
        max_num_batched_tokens=2048,
        gpu_memory_utilization=0.6,
        enforce_eager=True,
        enable_prefix_caching=False,
        enable_chunked_prefill=False,
        language_model_only=True,
        seed=42,
    )
    outputs = llm.generate(
        ["Calculate 1 + 1. The answer is", "Calculate 2 + 3. The answer is"],
        SamplingParams(n=4, temperature=1.0, top_p=1.0, top_k=-1, max_tokens=8),
    )
    assert len(outputs) == 2 and all(len(item.outputs) == 4 for item in outputs)
    for item in outputs:
        print({"prompt": item.prompt, "responses": [sample.text for sample in item.outputs]}, flush=True)
    print("QWEN38_VLLM_4NPU_ROLLOUT_OK", flush=True)


if __name__ == "__main__":
    main()
