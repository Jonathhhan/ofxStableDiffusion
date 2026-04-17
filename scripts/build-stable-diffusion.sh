#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ADDON_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SOURCE_DIR="$ADDON_ROOT/libs/stable-diffusion/source"
BUILD_DIR="$ADDON_ROOT/libs/stable-diffusion/build"
INSTALL_INCLUDE_DIR="$ADDON_ROOT/libs/stable-diffusion/include"
INSTALL_LIB_DIR="$ADDON_ROOT/libs/stable-diffusion/lib/Linux64"
JOBS="${JOBS:-}"
CLEAN=0
CPU_ONLY=0
CUDA=0
VULKAN=0
CONFIGURATION="Release"

write_step() {
	printf '==> %s\n' "$1"
}

die() {
	printf 'Error: %s\n' "$1" >&2
	exit 1
}

while [[ $# -gt 0 ]]; do
	case "$1" in
		--source-dir) SOURCE_DIR="$2"; shift 2 ;;
		--build-dir) BUILD_DIR="$2"; shift 2 ;;
		--install-include-dir) INSTALL_INCLUDE_DIR="$2"; shift 2 ;;
		--install-lib-dir) INSTALL_LIB_DIR="$2"; shift 2 ;;
		--jobs) JOBS="$2"; shift 2 ;;
		--config) CONFIGURATION="$2"; shift 2 ;;
		--clean) CLEAN=1; shift ;;
		--cpu-only) CPU_ONLY=1; shift ;;
		--cuda) CUDA=1; shift ;;
		--vulkan) VULKAN=1; shift ;;
		*)
			die "Unknown option: $1"
			;;
	esac
done

if [[ -z "$JOBS" ]]; then
	if command -v nproc >/dev/null 2>&1; then
		JOBS="$(nproc)"
	else
		JOBS=4
	fi
fi

command -v cmake >/dev/null 2>&1 || die "cmake was not found in PATH"
[[ -d "$SOURCE_DIR" ]] || die "stable-diffusion.cpp source is missing at $SOURCE_DIR"
[[ -f "$SOURCE_DIR/CMakeLists.txt" ]] || die "No CMakeLists.txt was found in $SOURCE_DIR"

if [[ "$CLEAN" -eq 1 ]]; then
	write_step "Cleaning previous stable-diffusion build"
	rm -rf "$BUILD_DIR"
fi

mkdir -p "$BUILD_DIR" "$INSTALL_INCLUDE_DIR" "$INSTALL_LIB_DIR"

CMAKE_ARGS=(
	-S "$SOURCE_DIR"
	-B "$BUILD_DIR"
	-DCMAKE_BUILD_TYPE="$CONFIGURATION"
	-DBUILD_SHARED_LIBS=ON
	-DSD_BUILD_SHARED_LIB=ON
	-DSD_BUILD_TESTS=OFF
	-DSD_BUILD_EXAMPLES=OFF
)

if [[ "$CPU_ONLY" -eq 1 ]]; then
	CMAKE_ARGS+=(-DGGML_CUDA=OFF -DGGML_VULKAN=OFF)
else
	CMAKE_ARGS+=(-DGGML_CUDA=$([[ "$CUDA" -eq 1 ]] && echo ON || echo OFF))
	CMAKE_ARGS+=(-DGGML_VULKAN=$([[ "$VULKAN" -eq 1 ]] && echo ON || echo OFF))
fi

write_step "Configuring stable-diffusion native library"
cmake "${CMAKE_ARGS[@]}"

write_step "Building stable-diffusion ($CONFIGURATION)"
cmake --build "$BUILD_DIR" --config "$CONFIGURATION" --parallel "$JOBS"

HEADER_PATH="$(find "$SOURCE_DIR" -name stable-diffusion.h | head -n 1 || true)"
DLL_PATH="$(find "$BUILD_DIR" -name 'libstable-diffusion.so' | head -n 1 || true)"

[[ -n "$HEADER_PATH" ]] || die "stable-diffusion.h was not found under $SOURCE_DIR"
[[ -n "$DLL_PATH" ]] || die "libstable-diffusion.so was not found under $BUILD_DIR"

write_step "Staging stable-diffusion artifacts into the addon"
cp -f "$HEADER_PATH" "$INSTALL_INCLUDE_DIR/"
cp -f "$DLL_PATH" "$INSTALL_LIB_DIR/"

printf '\nstable-diffusion native build complete.\n'
printf '  source:  %s\n' "$SOURCE_DIR"
printf '  build:   %s\n' "$BUILD_DIR"
printf '  include: %s\n' "$INSTALL_INCLUDE_DIR"
printf '  libs:    %s\n' "$INSTALL_LIB_DIR"
