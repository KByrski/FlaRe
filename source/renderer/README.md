# FlaRe CUDA Renderer

This directory contains the CUDA/OptiX renderer:

Build from the repository root:

```bash
./build.sh
```

The build uses the active Python environment and produces:

```text
source/renderer/output/PYOPTIXFLARERENDERER.so
source/renderer/output/shaders.cu.ptx
```

Both `source/train.py` and `source/evaluate.py` load the module from that
directory. The native module resolves `shaders.cu.ptx` relative to its own
location, so execution does not depend on the current working directory.
