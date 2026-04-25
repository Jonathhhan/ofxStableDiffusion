# Performance Benchmarks

This directory contains performance benchmark results and tracking tools.

## Running Benchmarks

### Quick Start

```bash
# Unix/Linux/macOS
./scripts/run_benchmarks.sh

# Windows
scripts\run_benchmarks.bat
```

### Manual Run

```bash
# Ensure tests are built
cmake -S tests -B tests/build -DCMAKE_BUILD_TYPE=Release
cmake --build tests/build --config Release

# Run benchmark script
python3 scripts/run_benchmarks.py
```

## Results

Benchmark results are saved to `benchmarks/results/` with timestamped filenames:
- `benchmark_YYYYMMDD_HHMMSS.json`

## Regression Detection

The benchmark runner automatically compares results against the previous run:

```
PERFORMANCE COMPARISON
============================================================

Unit Tests: ✓
  Current:  2.45s
  Baseline: 2.38s
  Delta:    +0.07s (+2.9%)
```

Warnings are shown for:
- **>10% change**: ⚠️ Warning indicator
- **>20% regression**: Explicit warning message
- **>20% improvement**: Celebration message ✨

## Adding Custom Benchmarks

Edit `scripts/run_benchmarks.py` to add new benchmarks:

```python
benchmarks = [
    {
        "name": "My Custom Benchmark",
        "command": "path/to/benchmark_executable --args"
    },
    # ...
]
```

## CI Integration

Benchmarks can run in CI via GitHub Actions:

1. **On every commit** - Quick benchmark of unit tests
2. **Scheduled nightly** - Full benchmark suite
3. **Manual dispatch** - Run benchmarks on demand

Results are saved as artifacts and can be downloaded.

## Performance Tracking

To track performance over time:

1. **Keep baseline results**: Don't delete old benchmark files
2. **Review trends**: Compare multiple runs to see trends
3. **Set thresholds**: Fail CI if regression > X%

## Benchmark Metrics

Current benchmarks track:
- Test execution time
- Success/failure rate
- System information

Future metrics could include:
- Memory usage
- CPU utilization
- Generation speed (images/second)
- Model loading time
- VRAM consumption
