#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
addon_root="$(cd "${script_dir}/.." && pwd)"
tests_root="${addon_root}/tests"
build_dir="${1:-${tests_root}/build}"

cmake -S "${tests_root}" -B "${build_dir}"
cmake --build "${build_dir}" --config Release
ctest --test-dir "${build_dir}" -C Release --output-on-failure
