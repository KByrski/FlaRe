#!/usr/bin/env bash

# Portable build for the CUDA/OptiX renderer integrated under source/renderer.
# Set OPTIX_DIRECTORY and, if needed, CUDA_DIRECTORY, PYTHON, CUDA_ARCH, and CXX.

set -Eeuo pipefail

PROJECT_DIRECTORY="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RENDERER_DIRECTORY="$PROJECT_DIRECTORY/source/renderer"
OUTPUT_DIRECTORY="$RENDERER_DIRECTORY/output"

CUDA_DIRECTORY="${CUDA_DIRECTORY:-${CUDA_HOME:-/usr/local/cuda}}"
OPTIX_DIRECTORY="${OPTIX_DIRECTORY:?Set OPTIX_DIRECTORY to the OptiX SDK root}"
PYTHON="${PYTHON:-python}"
CUDA_ARCH="${CUDA_ARCH:-89}"
CXX="${CXX:-g++}"
RUN_SMOKE_TEST="${RUN_SMOKE_TEST:-0}"

NVCC="$CUDA_DIRECTORY/bin/nvcc"
COMPUTE="compute_$CUDA_ARCH"
SM="sm_$CUDA_ARCH"

if [[ ! -x "$NVCC" ]]; then
    echo "Error: nvcc was not found at $NVCC" >&2
    exit 1
fi
if [[ ! -f "$OPTIX_DIRECTORY/include/optix.h" ]]; then
    echo "Error: OptiX headers were not found in $OPTIX_DIRECTORY/include" >&2
    exit 1
fi

TORCH_DIRECTORY="$("$PYTHON" -c 'import pathlib, torch; print(pathlib.Path(torch.__file__).resolve().parent)')"
PYTHON_INCLUDE="$("$PYTHON" -c 'import sysconfig; print(sysconfig.get_path("include"))')"
TORCH_ABI="$("$PYTHON" -c 'import torch; print(int(torch._C._GLIBCXX_USE_CXX11_ABI))')"

CUDA_INCLUDE="$CUDA_DIRECTORY/include"
if [[ -d "$CUDA_DIRECTORY/lib64" ]]; then
    CUDA_LIBRARY="$CUDA_DIRECTORY/lib64"
else
    CUDA_LIBRARY="$CUDA_DIRECTORY/lib"
fi
if [[ -d "$CUDA_LIBRARY/stubs" ]]; then
    CUDA_STUB_LIBRARY="$CUDA_LIBRARY/stubs"
elif [[ -d "$CUDA_DIRECTORY/targets/x86_64-linux/lib/stubs" ]]; then
    CUDA_STUB_LIBRARY="$CUDA_DIRECTORY/targets/x86_64-linux/lib/stubs"
else
    CUDA_STUB_LIBRARY="$CUDA_LIBRARY"
fi

TORCH_INCLUDE="$TORCH_DIRECTORY/include"
TORCH_API_INCLUDE="$TORCH_INCLUDE/torch/csrc/api/include"
TORCH_LIBRARY="$TORCH_DIRECTORY/lib"
OPTIX_INCLUDE="$OPTIX_DIRECTORY/include"

mkdir -p "$OUTPUT_DIRECTORY"

NVCC_FLAGS=(
    -std=c++17
    -Xcompiler -fPIC
    -arch="$COMPUTE"
    -code="$SM"
    -I"$OPTIX_INCLUDE"
    -I"$CUDA_INCLUDE"
)
CPP_FLAGS=(
    -std=c++17
    -fPIC
    -D_GLIBCXX_USE_CXX11_ABI="$TORCH_ABI"
    -I"$OPTIX_INCLUDE"
    -I"$CUDA_INCLUDE"
    -I"$TORCH_INCLUDE"
    -I"$TORCH_API_INCLUDE"
    -I"$PYTHON_INCLUDE"
)

echo "Building renderer with CUDA at $CUDA_DIRECTORY"
echo "OptiX SDK: $OPTIX_DIRECTORY"
echo "Python: $PYTHON"
echo "CUDA target: $COMPUTE / $SM"

"$NVCC" -std=c++17 -ptx -arch="$COMPUTE" \
    "$RENDERER_DIRECTORY/shaders.cu" \
    -o "$OUTPUT_DIRECTORY/shaders.cu.ptx" \
    -I"$OPTIX_INCLUDE"

CUDA_SOURCES=(
    CPyOptiXFLARERenderer_CUDA_Backward
    CPyOptiXFLARERenderer_CUDA_Backward_base
    CPyOptiXFLARERenderer_CUDA_Core
    CPyOptiXFLARERenderer_CUDA_Forward_training
    CPyOptiXFLARERenderer_CUDA_Forward_training_base
    CPyOptiXFLARERenderer_CUDA_Forward_inference
    CPyOptiXFLARERenderer_CUDA_Forward_inference_base
    CPyOptiXFLARERenderer_CUDA_GetMedianDepth
    CPyOptiXFLARERenderer_CUDA_GetMedianDepth_base
    GenerateRays
)

CUDA_OBJECTS=()
for source_name in "${CUDA_SOURCES[@]}"; do
    object="$OUTPUT_DIRECTORY/${source_name}.cu.o"
    "$NVCC" "${NVCC_FLAGS[@]}" \
        -c "$RENDERER_DIRECTORY/${source_name}.cu" \
        -o "$object"
    CUDA_OBJECTS+=("$object")
done

"$NVCC" -std=c++17 -Xcompiler -fPIC -dlink \
    "${CUDA_OBJECTS[@]}" \
    -o "$OUTPUT_DIRECTORY/dlink.o" \
    -arch="$COMPUTE" -code="$SM" \
    -L"$CUDA_STUB_LIBRARY" \
    -L"$CUDA_LIBRARY" \
    -lcuda -lcudart

"$CXX" "${CPP_FLAGS[@]}" \
    -c "$RENDERER_DIRECTORY/GenerateRays.cpp" \
    -o "$OUTPUT_DIRECTORY/GenerateRays.cpp.o"
"$CXX" "${CPP_FLAGS[@]}" \
    -c "$RENDERER_DIRECTORY/CPyOptiXFLARERenderer.cpp" \
    -o "$OUTPUT_DIRECTORY/CPyOptiXFLARERenderer.cpp.o"

"$CXX" "${CPP_FLAGS[@]}" -shared \
    "$RENDERER_DIRECTORY/PyOptiXFLARERenderer.cpp" \
    "${CUDA_OBJECTS[@]}" \
    "$OUTPUT_DIRECTORY/dlink.o" \
    "$OUTPUT_DIRECTORY/GenerateRays.cpp.o" \
    "$OUTPUT_DIRECTORY/CPyOptiXFLARERenderer.cpp.o" \
    -o "$OUTPUT_DIRECTORY/PYOPTIXFLARERENDERER.so" \
    -L"$CUDA_STUB_LIBRARY" \
    -L"$CUDA_LIBRARY" \
    -L"$TORCH_LIBRARY" \
    -lcuda -lcudart -ldl -ltorch_python -ltorch -ltorch_cpu -lc10 \
    -Wl,-rpath,"$CUDA_LIBRARY" \
    -Wl,-rpath,"$TORCH_LIBRARY"

echo "Build completed: $OUTPUT_DIRECTORY/PYOPTIXFLARERENDERER.so"

if [[ "$RUN_SMOKE_TEST" == "1" ]]; then
    "$PYTHON" "$PROJECT_DIRECTORY/scripts/smoke_test_renderer.py"
fi
