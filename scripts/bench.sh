#!/usr/bin/env bash
# Run the self-benchmark suite and write reports into ./bench-out/.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p bench-out
PYTHONPATH=src python3 -c "
from llmdiff.benchmark.cli_hooks import bench
out = bench(out_md='bench-out/benchmark.md', out_json='bench-out/benchmark.json', db='bench-out/bench.db', label='local')
print('detection rate:', out['metrics']['detection_rate'])
print('reports written to bench-out/')
"
