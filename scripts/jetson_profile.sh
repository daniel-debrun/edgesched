#!/usr/bin/env bash
# Capture latency profiles for TensorRT engines on a Jetson and write edgesched JSON.
#
# Not yet run: there are no Jetson measurements so far, and the RTX 4080 runs used
# the TensorRT pip wheel, which does not ship trtexec.
#
# Usage (on the device, JetPack with TensorRT installed):
#   scripts/jetson_profile.sh model.onnx model_name "input:1x3x224x224" 1,2,4,8 [out_dir]
#
# The input spec's leading dim is replaced by each batch size. Engines are built with
# a dynamic-batch optimization profile covering the full range, then timed per batch
# size with trtexec. Put the device in a fixed power mode first and record it:
#   sudo nvpmodel -m 0 && sudo jetson_clocks
set -euo pipefail

ONNX=${1:?onnx path}
NAME=${2:?model name}
INPUT=${3:?input spec, e.g. input:1x3x224x224}
BATCHES=${4:-1,2,4,8}
OUT=${5:-profiles}
TRTEXEC=${TRTEXEC:-/usr/src/tensorrt/bin/trtexec}

mkdir -p "$OUT/trtexec"
IN_NAME=${INPUT%%:*}
DIMS=${INPUT#*:}
REST=${DIMS#*x}
IFS=, read -ra BS <<< "$BATCHES"
MAXB=${BS[-1]}
ENGINE="$OUT/trtexec/${NAME}.engine"

"$TRTEXEC" --onnx="$ONNX" --saveEngine="$ENGINE" --fp16 \
  --minShapes="${IN_NAME}:1x${REST}" --optShapes="${IN_NAME}:${MAXB}x${REST}" \
  --maxShapes="${IN_NAME}:${MAXB}x${REST}" > "$OUT/trtexec/${NAME}_build.log"

for B in "${BS[@]}"; do
  "$TRTEXEC" --loadEngine="$ENGINE" --shapes="${IN_NAME}:${B}x${REST}" \
    --warmUp=500 --duration=20 --useSpinWait \
    --exportTimes="$OUT/trtexec/${NAME}_b${B}.json" > "$OUT/trtexec/${NAME}_b${B}.log"
done

# Convert per-inference times (ms) to an edgesched profile. Uses the p50 as the
# point estimate and keeps p90/p99 for choosing a safety factor.
python3 - "$NAME" "$OUT" "${BS[@]}" <<'PY'
import json, statistics, sys
name, out, *batches = sys.argv[1:]
points, stats = {}, {}
for b in batches:
    times = sorted(t["latencyMs"] for t in json.load(open(f"{out}/trtexec/{name}_b{b}.json")))
    q = lambda p: times[min(len(times) - 1, int(p * (len(times) - 1)))]
    points[b] = q(0.5)
    stats[b] = {"mean": statistics.fmean(times), "p50": q(0.5), "p90": q(0.9), "p99": q(0.99),
                "max": times[-1]}
json.dump({"model": name, "unit": "ms", "quantile": 0.5, "points": points, "stats": stats,
           "meta": {"source": "trtexec", "note": "latencyMs includes H2D/D2H copies"}},
          open(f"{out}/{name}.json", "w"), indent=2)
print(f"wrote {out}/{name}.json")
PY

# Then reference it from a workload:
#   models:
#     seg: {max_batch: 8, batchable: true, profile: {type: json, path: ../profiles/seg.json}}
