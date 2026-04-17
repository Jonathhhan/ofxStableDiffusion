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
AUTO=1
CPU_ONLY=0
CUDA=0
VULKAN=0
METAL=0
CONFIGURATION="Release"

write_step() {
	printf '==> %s\n' "$1"
}

die() {
	printf 'Error: %s\n' "$1" >&2
	exit 1
}

detect_cuda() {
	[[ -n "${CUDA_PATH:-}" && -d "${CUDA_PATH:-}" ]] && return 0
	command -v nvcc >/dev/null 2>&1
}

detect_vulkan() {
	[[ -n "${VULKAN_SDK:-}" && -d "${VULKAN_SDK:-}" ]] && return 0
	command -v glslc >/dev/null 2>&1 || command -v vulkaninfo >/dev/null 2>&1
}

detect_metal() {
	[[ "$(uname -s 2>/dev/null || echo unknown)" == "Darwin" ]] || return 1
	command -v xcrun >/dev/null 2>&1
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
		--cpu|--cpu-only) CPU_ONLY=1; CUDA=0; VULKAN=0; METAL=0; AUTO=0; shift ;;
		--gpu|--cuda) CPU_ONLY=0; CUDA=1; AUTO=0; shift ;;
		--vulkan) CPU_ONLY=0; VULKAN=1; AUTO=0; shift ;;
		--metal) CPU_ONLY=0; METAL=1; AUTO=0; shift ;;
		--auto) AUTO=1; CPU_ONLY=0; CUDA=0; VULKAN=0; METAL=0; shift ;;
		--help|-h)
			cat <<'EOF'
build-stable-diffusion.sh - Build the bundled stable-diffusion.cpp runtime.

Usage:
  ./scripts/build-stable-diffusion.sh [OPTIONS]

Options:
  --cpu, --cpu-only      Build CPU backend only
  --auto                 Auto-detect GPU backends (default)
  --gpu, --cuda          Enable CUDA backend
  --vulkan               Enable Vulkan backend
  --metal                Enable Metal backend (macOS only)
  --jobs N               Parallel build jobs
  --config NAME          Build configuration (default: Release)
  --clean                Remove previous build directory before building
  --source-dir DIR       Override vendored source directory
  --build-dir DIR        Override build directory
  --install-lib-dir DIR  Override staged library directory
  --help                 Show this help message
EOF
			exit 0
			;;
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

ENABLE_CUDA=0
ENABLE_VULKAN=0
ENABLE_METAL=0
BACKEND_MODE="auto-detect"

if [[ "$CPU_ONLY" -eq 1 ]]; then
	BACKEND_MODE="cpu-only"
elif [[ "$CUDA" -eq 1 || "$VULKAN" -eq 1 || "$METAL" -eq 1 ]]; then
	BACKEND_MODE="explicit"
	ENABLE_CUDA="$CUDA"
	ENABLE_VULKAN="$VULKAN"
	ENABLE_METAL="$METAL"
else
	if detect_cuda; then
		ENABLE_CUDA=1
	fi
	if detect_vulkan; then
		ENABLE_VULKAN=1
	fi
	if detect_metal; then
		ENABLE_METAL=1
	fi
fi

CMAKE_ARGS=(
	-S "$SOURCE_DIR"
	-B "$BUILD_DIR"
	-DCMAKE_BUILD_TYPE="$CONFIGURATION"
	-DSD_BUILD_SHARED_LIBS=ON
	-DSD_BUILD_EXAMPLES=OFF
	-DSD_CUDA=$([[ "$ENABLE_CUDA" -eq 1 ]] && echo ON || echo OFF)
	-DSD_VULKAN=$([[ "$ENABLE_VULKAN" -eq 1 ]] && echo ON || echo OFF)
	-DSD_METAL=$([[ "$ENABLE_METAL" -eq 1 ]] && echo ON || echo OFF)
)

write_step "Configuring stable-diffusion native library"
printf '    Backend mode: %s (CUDA=%s, Vulkan=%s, Metal=%s)\n' \
	"$BACKEND_MODE" \
	"$([[ "$ENABLE_CUDA" -eq 1 ]] && echo ON || echo OFF)" \
	"$([[ "$ENABLE_VULKAN" -eq 1 ]] && echo ON || echo OFF)" \
	"$([[ "$ENABLE_METAL" -eq 1 ]] && echo ON || echo OFF)"
cmake "${CMAKE_ARGS[@]}"

write_step "Building stable-diffusion ($CONFIGURATION)"
cmake --build "$BUILD_DIR" --config "$CONFIGURATION" --parallel "$JOBS"

HEADER_PATH="$(find "$SOURCE_DIR" -name stable-diffusion.h | head -n 1 || true)"
DLL_PATH="$(find "$BUILD_DIR" \( -name 'libstable-diffusion.so' -o -name 'libstable-diffusion.dylib' \) | head -n 1 || true)"

[[ -n "$HEADER_PATH" ]] || die "stable-diffusion.h was not found under $SOURCE_DIR"
[[ -n "$DLL_PATH" ]] || die "stable-diffusion runtime library was not found under $BUILD_DIR"

write_step "Staging stable-diffusion artifacts into the addon"
cp -f "$DLL_PATH" "$INSTALL_LIB_DIR/"

printf '\nstable-diffusion native build complete.\n'
printf '  source:  %s\n' "$SOURCE_DIR"
printf '  build:   %s\n' "$BUILD_DIR"
printf '  include: %s\n' "$INSTALL_INCLUDE_DIR"
printf '  libs:    %s\n' "$INSTALL_LIB_DIR"
