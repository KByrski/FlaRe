# Optional OptiX viewer

The optional low-level ray tracer displays a trained FlaRe scene together with
user-supplied Stanford meshes. Scene-specific assets are not included.

The viewer is isolated from FlaRe training: it has a separate Python entry point,
CUDA/OptiX sources, PTX file, and shared library. Building it does not replace
`source/renderer/output/PYOPTIXFLARERENDERER.so`, and `source/train.py` does not
import viewer code.

## Build

Use the same CUDA 12.4, PyTorch environment, and OptiX SDK as the main renderer:

```bash
OPTIX_DIRECTORY="/path/to/NVIDIA-OptiX-SDK-8.x" \
PYTHON=/path/to/flare-public-cu124/bin/python \
CUDA_ARCH=89 \
./build_viewer.sh
```

The viewer targets RTX 4090 (`CUDA_ARCH=89`). Blackwell support is planned but
has not yet been validated for this optional ray tracer.

## Stanford meshes

Download the bunny or dragon from the Stanford 3D Scanning Repository and pass
the extracted PLY explicitly. Models are not redistributed by this repository.
Typical archive members are `bun_zipper.ply` and `dragon_vrip.ply`.

## Headless render

```bash
python source/viewer.py \
  --scene_path /path/to/garden \
  --checkpoint /path/to/best.checkpoint \
  --resolution 4 \
  --mesh /path/to/dragon_vrip.ply \
  --scale 0.01 \
  --translation X Y Z \
  --material glass \
  --ior 1.12 \
  --supersample 2 \
  --output viewer_render.png
```

`--translation` is expressed in the FlaRe scene's world coordinates. Multiple
`--mesh` arguments are supported; the initial implementation applies the same
transform and material to all supplied objects.

## Interactive mode

Add `--interactive` on a node with a graphical display. Controls are W/A/S/D,
Space/C for vertical movement, arrow keys for camera rotation, R to save the
current frame, and Q or Escape to exit. Headless mode remains useful through SSH
and for reproducible screenshots.

The default glass IOR is 1.12, matching the accepted Garden visualization;
pass `--ior` explicitly for other materials. `--supersample N` renders at N times
the output dimensions and downsamples with Lanczos.

Important controls include `--color R G B`, `--ior`, `--light_direction X Y Z`,
`--ambient R G B`, `--shadow_multiplier`, and `--recursion_depth`.
