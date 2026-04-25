#!/bin/bash
# Run performance benchmarks and track regressions

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$REPO_ROOT"

echo "Performance Benchmark Runner"
echo "============================"
echo ""

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "Error: python3 is required for benchmarking"
    exit 1
fi

# Check if tests are built
if [ ! -d "tests/build" ]; then
    echo "Building tests first..."
    cmake -S tests -B tests/build -DCMAKE_BUILD_TYPE=Release
    cmake --build tests/build --config Release
fi

# Run benchmark script
python3 scripts/run_benchmarks.py "$@"

echo ""
echo "Benchmarks complete!"
echo "Results saved to: benchmarks/results/"
