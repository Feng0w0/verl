#!/usr/bin/env bash
# Usage:
#   bash start_qwen38_full48_4node_ray.sh head
#   bash start_qwen38_full48_4node_ray.sh worker 172.27.2.128
set -euo pipefail

ROLE=${1:-}
NODE_IP=${2:-}
HEAD_IP=172.27.2.132
HEAD_PORT=26379
ROOT=/mnt/share/h00943455/Qwen3.8-Flash-Next
CONDA_ENV=/mnt/share/h00943455/conda_envs/verl-qwen3.8-flash-veomni
CANN_ROOT=/mnt/share/cann/CANN_9_2_0_B020

usage() {
  echo "Usage: $0 head | worker <172.27.2.128|172.27.2.131|172.27.2.132>" >&2
  exit 2
}

case "$ROLE" in
  head) NODE_IP=$HEAD_IP ;;
  worker) [[ -n "$NODE_IP" ]] || usage ;;
  *) usage ;;
esac


LOCAL_IP=$(ip -4 -o addr show dev data0.172 | awk '{print $4}' | cut -d/ -f1)
[[ "$LOCAL_IP" == "$NODE_IP" ]] || {
  echo "This host has data0.172 address $LOCAL_IP, expected $NODE_IP" >&2
  exit 1
}

export PATH="$CONDA_ENV/bin:$PATH" CONDA_PREFIX="$CONDA_ENV"
set +u
source "$CANN_ROOT/cann/set_env.sh"
source "$CANN_ROOT/nnal/atb/set_env.sh" --cxx_abi=1
set -u
export LD_PRELOAD="$CONDA_ENV/lib/libstdc++.so.6${LD_PRELOAD:+:$LD_PRELOAD}"
export PYTHONPATH="$ROOT/vllm-ascend:$ROOT/VeOmni:$ROOT/verl:${PYTHONPATH:-}"
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
export HCCL_SOCKET_IFNAME=data0.172 GLOO_SOCKET_IFNAME=data0.172 HCCL_IF_IP="$LOCAL_IP"
export RAY_USAGE_STATS_ENABLED=0
export VLLM_ASCEND_SYNC_BEFORE_CAMEM_SLEEP=1

COMMON=(
  --node-ip-address="$NODE_IP"
  --num-cpus=96
  --resources='{"NPU":16}'
  --min-worker-port=22000
  --max-worker-port=22999
  --disable-usage-stats
)

if [[ "$ROLE" == head ]]; then
  ray start --head --port="$HEAD_PORT" --temp-dir=/tmp/ray-qwen38-full48-4node "${COMMON[@]}"
else
  ray start --address="$HEAD_IP:$HEAD_PORT" "${COMMON[@]}"
fi

ray status --address="$HEAD_IP:$HEAD_PORT"
