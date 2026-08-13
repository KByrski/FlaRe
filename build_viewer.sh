#!/usr/bin/env bash

# Build the optional inference-only OptiX viewer. The training renderer under
# source/renderer is deliberately not touched.
set -Eeuo pipefail

PROJECT_DIRECTORY="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RENDERER_DIRECTORY="$PROJECT_DIRECTORY/source/viewer_renderer"
OUTPUT_DIRECTORY="$RENDERER_DIRECTORY/output"
if [[ -z "${CUDA_DIRECTORY:-}" ]]; then
    if [[ -n "${CUDA_HOME:-}" ]]; then
        CUDA_DIRECTORY="$CUDA_HOME"
    elif command -v nvcc >/dev/null 2>&1; then
        CUDA_DIRECTORY="$(cd -- "$(dirname -- "$(command -v nvcc)")/.." && pwd)"
    else
        CUDA_DIRECTORY=/usr/local/cuda
    fi
fi
OPTIX_DIRECTORY="${OPTIX_DIRECTORY:?Set OPTIX_DIRECTORY to the OptiX SDK root}"
PYTHON="${PYTHON:-python}"
CUDA_ARCH="${CUDA_ARCH:-89}"
CXX="${CXX:-g++}"
NVCC="$CUDA_DIRECTORY/bin/nvcc"
COMPUTE="compute_$CUDA_ARCH"
SM="sm_$CUDA_ARCH"

[[ -x "$NVCC" ]] || { echo "Error: nvcc was not found at $NVCC" >&2; exit 1; }
[[ -f "$OPTIX_DIRECTORY/include/optix.h" ]] || {
    echo "Error: OptiX headers were not found in $OPTIX_DIRECTORY/include" >&2
    exit 1
}

TORCH_DIRECTORY="$("$PYTHON" -c 'import pathlib, torch; print(pathlib.Path(torch.__file__).resolve().parent)')"
PYTHON_INCLUDE="$("$PYTHON" -c 'import sysconfig; print(sysconfig.get_path("include"))')"
TORCH_ABI="$("$PYTHON" -c 'import torch; print(int(torch._C._GLIBCXX_USE_CXX11_ABI))')"
CUDA_INCLUDE="$CUDA_DIRECTORY/include"
if [[ -d "$CUDA_DIRECTORY/lib64" ]]; then CUDA_LIBRARY="$CUDA_DIRECTORY/lib64"; else CUDA_LIBRARY="$CUDA_DIRECTORY/lib"; fi
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

NVCC_FLAGS=(-std=c++17 -Xcompiler -fPIC -arch="$COMPUTE" -code="$SM" -I"$OPTIX_INCLUDE" -I"$CUDA_INCLUDE")
CPP_FLAGS=(-std=c++17 -fPIC -D_GLIBCXX_USE_CXX11_ABI="$TORCH_ABI" -I"$OPTIX_INCLUDE" -I"$CUDA_INCLUDE" -I"$TORCH_INCLUDE" -I"$TORCH_API_INCLUDE" -I"$PYTHON_INCLUDE")

echo "Building optional viewer for $COMPUTE / $SM"
"$NVCC" -std=c++17 -ptx -arch="$COMPUTE" "$RENDERER_DIRECTORY/shaders.cu" \
    -o "$OUTPUT_DIRECTORY/viewer_shaders.cu.ptx" -I"$OPTIX_INCLUDE"

CUDA_SOURCES=(CPyOptiXFLAREVIEWERRenderer_CUDA_Core CPyOptiXFLAREVIEWERRenderer_CUDA_Forward GenerateRays)
CUDA_OBJECTS=()
for source_name in "${CUDA_SOURCES[@]}"; do
    object="$OUTPUT_DIRECTORY/${source_name}.cu.o"
    "$NVCC" "${NVCC_FLAGS[@]}" -c "$RENDERER_DIRECTORY/${source_name}.cu" -o "$object"
    CUDA_OBJECTS+=("$object")
done
"$NVCC" -std=c++17 -Xcompiler -fPIC -dlink "${CUDA_OBJECTS[@]}" \
    -o "$OUTPUT_DIRECTORY/dlink.o" -arch="$COMPUTE" -code="$SM" \
    -L"$CUDA_STUB_LIBRARY" -L"$CUDA_LIBRARY" -lcuda -lcudart

CPP_SOURCES=(GenerateRays CPyOptiXFLAREVIEWERMeshInstance CPyOptiXFLAREVIEWERRenderer)
CPP_OBJECTS=()
for source_name in "${CPP_SOURCES[@]}"; do
    object="$OUTPUT_DIRECTORY/${source_name}.cpp.o"
    "$CXX" "${CPP_FLAGS[@]}" -c "$RENDERER_DIRECTORY/${source_name}.cpp" -o "$object"
    CPP_OBJECTS+=("$object")
done
"$CXX" "${CPP_FLAGS[@]}" -shared "$RENDERER_DIRECTORY/PyOptiXFLAREVIEWERRenderer.cpp" \
    "${CUDA_OBJECTS[@]}" "$OUTPUT_DIRECTORY/dlink.o" "${CPP_OBJECTS[@]}" \
    -o "$OUTPUT_DIRECTORY/PYOPTIXFLAREVIEWER.so" \
    -L"$CUDA_STUB_LIBRARY" -L"$CUDA_LIBRARY" -L"$TORCH_LIBRARY" \
    -lcuda -lcudart -ldl -ltorch_python -ltorch -ltorch_cpu -lc10 \
    -Wl,-rpath,"$CUDA_LIBRARY" -Wl,-rpath,"$TORCH_LIBRARY"
echo "Viewer build completed: $OUTPUT_DIRECTORY/PYOPTIXFLAREVIEWER.so"
