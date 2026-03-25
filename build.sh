#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${ROOT_DIR}/build_2d"
THREADS="${THREADS:-$(nproc)}"

# Ensure pybind11 is present for the pybind module build.
if [[ ! -d "${ROOT_DIR}/pybind11" ]]; then
  echo "[build.sh] Cloning pybind11..."
  git clone https://github.com/pybind/pybind11.git "${ROOT_DIR}/pybind11"
fi

# Remove existing build directory if it exists
if [[ -d "${BUILD_DIR}" ]]; then
  echo "[build.sh] Removing existing build directory ${BUILD_DIR}"
  rm -rf "${BUILD_DIR}"
fi

# Use a temporary source tree so we can build from CMakeLists_xbar_2d.txt
# without modifying the repository's top-level CMakeLists.txt.
TMP_SRC_DIR="$(mktemp -d)"
cleanup() {
  rm -rf "${TMP_SRC_DIR}"
}
trap cleanup EXIT

cp "${ROOT_DIR}/CMakeLists_xbar_2d.txt" "${TMP_SRC_DIR}/CMakeLists.txt"
ln -s "${ROOT_DIR}/src" "${TMP_SRC_DIR}/src"
ln -s "${ROOT_DIR}/include" "${TMP_SRC_DIR}/include"
ln -s "${ROOT_DIR}/pybind11" "${TMP_SRC_DIR}/pybind11"
ln -s "${ROOT_DIR}/python" "${TMP_SRC_DIR}/python"

echo "[build.sh] Configuring 2D wrapper build in ${BUILD_DIR}"
cmake -S "${TMP_SRC_DIR}" -B "${BUILD_DIR}" -DCMAKE_BUILD_TYPE=Release

echo "[build.sh] Building xbar_simulator_2d with ${THREADS} threads"
cmake --build "${BUILD_DIR}" --parallel "${THREADS}"

echo "[build.sh] Done. Python module is expected under ${ROOT_DIR}/python"
