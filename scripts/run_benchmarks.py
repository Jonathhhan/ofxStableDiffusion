#!/usr/bin/env python3
"""
Performance Benchmark Runner

Runs performance benchmarks and tracks regressions over time.
"""

import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

class BenchmarkRunner:
    def __init__(self, results_dir="benchmarks/results"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def run_benchmark(self, test_name, command):
        """Run a single benchmark test."""
        print(f"Running benchmark: {test_name}")
        print(f"Command: {command}")

        start_time = datetime.now()
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=600  # 10 minute timeout
            )
            end_time = datetime.now()
            elapsed = (end_time - start_time).total_seconds()

            return {
                "name": test_name,
                "command": command,
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "elapsed_seconds": elapsed,
                "timestamp": start_time.isoformat(),
                "success": result.returncode == 0
            }
        except subprocess.TimeoutExpired:
            return {
                "name": test_name,
                "command": command,
                "returncode": -1,
                "error": "Timeout",
                "elapsed_seconds": 600,
                "timestamp": start_time.isoformat(),
                "success": False
            }

    def save_results(self, results, filename=None):
        """Save benchmark results to JSON file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"benchmark_{timestamp}.json"

        filepath = self.results_dir / filename
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\nResults saved to: {filepath}")
        return filepath

    def compare_results(self, current_file, baseline_file=None):
        """Compare current results against baseline."""
        with open(current_file) as f:
            current = json.load(f)

        if baseline_file is None:
            # Find most recent previous result
            results_files = sorted(self.results_dir.glob("benchmark_*.json"))
            if len(results_files) < 2:
                print("No baseline found for comparison")
                return

            baseline_file = results_files[-2]

        with open(baseline_file) as f:
            baseline = json.load(f)

        print("\n" + "="*60)
        print("PERFORMANCE COMPARISON")
        print("="*60)

        for curr_test in current["tests"]:
            test_name = curr_test["name"]
            curr_time = curr_test.get("elapsed_seconds", 0)

            # Find matching baseline test
            baseline_test = next(
                (t for t in baseline["tests"] if t["name"] == test_name),
                None
            )

            if baseline_test is None:
                print(f"\n{test_name}: NEW TEST")
                print(f"  Time: {curr_time:.2f}s")
                continue

            base_time = baseline_test.get("elapsed_seconds", 0)
            if base_time == 0:
                continue

            diff = curr_time - base_time
            diff_pct = (diff / base_time) * 100

            status = "⚠️" if abs(diff_pct) > 10 else "✓"

            print(f"\n{test_name}: {status}")
            print(f"  Current:  {curr_time:.2f}s")
            print(f"  Baseline: {base_time:.2f}s")
            print(f"  Delta:    {diff:+.2f}s ({diff_pct:+.1f}%)")

            if diff_pct > 20:
                print(f"  ⚠️  WARNING: Significant regression!")
            elif diff_pct < -20:
                print(f"  ✨ Significant improvement!")

def main():
    runner = BenchmarkRunner()

    # Define benchmarks
    benchmarks = [
        {
            "name": "Unit Tests",
            "command": "cmake --build tests/build --config Release && ctest --test-dir tests/build -C Release"
        },
        # Add more benchmarks as needed
    ]

    print("="*60)
    print("PERFORMANCE BENCHMARK SUITE")
    print("="*60)

    results = {
        "timestamp": datetime.now().isoformat(),
        "system": {
            "platform": sys.platform,
            "python_version": sys.version,
        },
        "tests": []
    }

    # Run each benchmark
    for bench in benchmarks:
        result = runner.run_benchmark(bench["name"], bench["command"])
        results["tests"].append(result)

        if result["success"]:
            print(f"  ✓ Completed in {result['elapsed_seconds']:.2f}s")
        else:
            print(f"  ✗ Failed")

    # Save results
    results_file = runner.save_results(results)

    # Compare against baseline
    runner.compare_results(results_file)

    # Exit with error if any test failed
    if not all(t["success"] for t in results["tests"]):
        sys.exit(1)

if __name__ == "__main__":
    main()
