<div align="center">

<img src="assets/flare_logo.gif" width="55%" alt="FlaRe"/>

# Floating Radiance Networks

**Explicit ray-traceable geometry with continuous local neural radiance fields**

<p>
<a href="https://scholar.google.com/citations?user=ef2YKtsAAAAJ&hl=pl&oi=sra">Krzysztof Byrski<sup>1</sup></a>,
<a href="https://github.com/rtobiasz">Rafał Tobiasz<sup>1,3</sup></a>,
<a href="https://grzegorzwilczynski.com">Grzegorz Wilczyński<sup>1,3</sup></a>,
<a href="https://github.com/MikolajZielinski">Mikołaj Zieliński<sup>2</sup></a>,
Dawid Baran,<br>
<a href="https://scholar.google.com/citations?user=3XvUbbMAAAAJ&hl=en">Dominik Belter<sup>2</sup></a>,
<a href="https://scholar.google.pl/citations?user=zSKYziUAAAAJ&hl=pl">Jacek Tabor<sup>1</sup></a>,
<a href="https://scholar.google.com/citations?user=0kp0MbgAAAAJ&hl=en">Przemysław Spurek<sup>1,3</sup></a>
</p>

<p>
<a href="https://en.uj.edu.pl/en"><sup>1</sup>Jagiellonian University</a> &nbsp;
<a href="https://put.poznan.pl/en"><sup>2</sup>Poznań University of Technology</a> &nbsp;
<a href="https://www.ideas.edu.pl/en/"><sup>3</sup>IDEAS Research Institute</a>
</p>

<p>
  <a href="https://arxiv.org/abs/2608.05920"><img src="https://img.shields.io/badge/arXiv-2608.05920-b31b1b.svg" alt="arXiv"></a>
  <a href="https://kbyrski.github.io/FlaRe"><img src="https://img.shields.io/badge/🌐-Project%20Page-blue" alt="Project Page"></a>
</p>

</div>

<p align="center">
  <img src="assets/teaser.png" width="100%" alt="FlaRe teaser: reconstruction, editing, style transfer, mesh extraction, and ray tracing"/>
</p>

<p align="center">
  <img src="assets/edit_animation.gif" width="33.333%" alt="Primitive-level geometry editing"/><img src="assets/style_transfer.gif" width="33.333%" alt="Image-guided appearance style transfer"/><img src="assets/raytracing.gif" width="33.333%" alt="Recursive ray tracing with inserted glass objects"/>
</p>

## Overview

FlaRe represents a scene with floating planar generalized Gaussian primitives.
Each primitive stores a compact descriptor of a local radiance field, while a
lightweight auto-decoder shared by the scene maps the descriptor, local surface
coordinates, and viewing direction to color and opacity.

The explicit representation is directly ray-queryable through OptiX. The same
scene model supports neural rendering and conventional ray-based graphics
operations without conversion to a separate representation.

## Repository layout

```text
.
├── build.sh                         # CUDA/OptiX renderer build
├── environment.yml                 # Reproducible Conda environment
├── requirements.txt                # Exact Python package versions
├── scripts/
│   ├── smoke_test_renderer.py       # Native renderer smoke test
│   ├── calculate_model_size.py      # Inference-model size
│   ├── encoding_2d_benchmark.py     # LUT-Encoding ablation
│   ├── run_mipnerf360.sh            # Mip-NeRF360 benchmark
│   ├── run_deep_blending.sh         # Deep Blending benchmark
│   └── run_tandt.sh                 # Tanks and Temples benchmark
└── source/
    ├── train.py                     # Training
    ├── evaluate.py                  # Best-checkpoint evaluation
    ├── extract_mesh.py              # Depth/normal rendering and TSDF fusion
    ├── render_style.py              # CLIP/VGG appearance stylization
    ├── style_transfer.py            # Style objectives and optimization
    ├── scene/                       # Scene loading and FlaRe parameters
    └── renderer/                    # CUDA/OptiX implementation
```

## Installation

### Requirements

The reference environment uses:

- Linux x86-64;
- NVIDIA driver 610.43.02 (CUDA 12.4 requires at least 550.54.15 on Linux);
- Python 3.11.6;
- PyTorch 2.6.0 with CUDA 12.4;
- CUDA Toolkit 12.4.1 (`nvcc` 12.4.131);
- NVIDIA OptiX SDK 8.0.0;
- a C++17 compiler (tested with GCC 13.3.0);
- an NVIDIA GPU supported by OptiX.

An RTX GPU is recommended for hardware-accelerated ray tracing. The default
build target is compute capability 8.9, matching an RTX 4090. Set `CUDA_ARCH`
when building for a different GPU.

> **Not tested on Blackwell GPUs.** Blackwell support is planned.

CUDA 12.4 Update 1 requires a sufficiently recent NVIDIA driver. See the
[CUDA 12.4 release notes](https://docs.nvidia.com/cuda/archive/12.4.1/cuda-toolkit-release-notes/index.html)
for the official compatibility table.

### Create the environment

Run the following commands from the repository root:

```bash
conda env create -f environment.yml
conda activate flare
```

The environment installs the exact Python versions listed in
[`requirements.txt`](requirements.txt) and the CUDA 12.4.1 development
toolchain required to compile the native extensions.

Build and install the local `simple-knn` CUDA extension:

```bash
CUDA_HOME="$CONDA_PREFIX" \
TORCH_CUDA_ARCH_LIST=8.9 \
pip install --no-build-isolation ./source/submodules/simple-knn
```

Set `TORCH_CUDA_ARCH_LIST` to the compute capability of the target GPU when
building for a supported architecture other than the RTX 4090.

### Install OptiX

Download **NVIDIA OptiX SDK 8.0.0** for Linux from the
[NVIDIA OptiX legacy downloads](https://developer.nvidia.com/designworks/optix/downloads/legacy).
The download requires a free NVIDIA Developer Program account.

Extract the SDK outside the repository and point `OPTIX_DIRECTORY` to the
directory containing `include/optix.h`:

```bash
export OPTIX_DIRECTORY=/path/to/NVIDIA-OptiX-SDK-8.0.0-linux64-x86_64
test -f "$OPTIX_DIRECTORY/include/optix.h"
```

OptiX supplies headers and device-side support; it is not installed through
Conda or pip.

### Build the renderer

Build the PyTorch/OptiX extension from the repository root:

```bash
CUDA_DIRECTORY="$CONDA_PREFIX" \
OPTIX_DIRECTORY="$OPTIX_DIRECTORY" \
CUDA_ARCH=89 \
./build.sh
```

Common architecture values are:

| GPU example | `CUDA_ARCH` |
|---|---:|
| A100 | `80` |
| RTX 3090 / A6000 | `86` |
| RTX 4090 | `89` |

The build produces:

```text
source/renderer/output/PYOPTIXFLARERENDERER.so
source/renderer/output/shaders.cu.ptx
```

On an RTX 4090, the build and renderer smoke test can be combined:

```bash
CUDA_DIRECTORY="$CONDA_PREFIX" \
OPTIX_DIRECTORY="$OPTIX_DIRECTORY" \
CUDA_ARCH=89 \
RUN_SMOKE_TEST=1 \
./build.sh
```

The standalone equivalent is:

```bash
python scripts/smoke_test_renderer.py
```

## Data

Training accepts either:

- a COLMAP scene containing `sparse/` and `images/`; or
- a Blender-format scene containing `transforms_train.json`.

The paper benchmarks use
[Mip-NeRF360](https://jonbarron.info/mipnerf360/) and the Tanks and Temples /
Deep Blending scenes distributed with
[3D Gaussian Splatting](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/input/tandt_db.zip).

## Training

Run training from the repository root:

```bash
python source/train.py \
    --source_path /path/to/scene \
    --resolution 2
```

This command uses all default optimization parameters. New runs are written to
the next available numeric directory:

```text
output/<run-id>/
├── checkpoints/
│   ├── best.checkpoint
│   ├── last.checkpoint
│   └── checkpoint_metadata.json
├── renders/
└── stats/
```

Training keeps rolling `best` and `last` checkpoints rather than every
iteration. Once the full FlaRe model becomes active, `best.checkpoint` is
selected using test-set FlaRe PSNR. At the end of training, the best checkpoint
is rendered automatically.

Use `--help` after building the renderer to list all parameters:

```bash
python source/train.py --help
```

### Geometry-aware training

Meaningful TSDF reconstruction requires a checkpoint trained with the 2DGS
depth-distortion and normal-consistency regularizers. Use the following configuration for depth, normal, and mesh extraction:

```bash
python source/train.py \
    --source_path /path/to/scene \
    --resolution 2 \
    --reg_depth_lambda 0.001 \
    --reg_depth_start_iter 3000 \
    --reg_normal_lambda 0.05 \
    --reg_normal_start_iter 7000 \
    --reg_normal_ramp_iters 3000 \
    --reg_normal_depth_edge_threshold 0.02
```

The normal term needs neighboring pixels, so training samples one complete
camera raster per step and randomizes camera order between epochs. Normal
regularization is disabled by default (`--reg_normal_lambda 0`).

### Depth, normals, and TSDF mesh

Render expected depth and both 2DGS normal products from the training cameras,
then fuse the RGB-D views into an Open3D TSDF volume:

```bash
python source/extract_mesh.py \
    --scene_path /path/to/scene \
    --checkpoint output/<run-id>/checkpoints/best.checkpoint \
    --renderer flare \
    --depth_mode expected \
    --resolution 2 \
    --save_renders
```

The output contains the raw and component-filtered meshes, `mesh_metadata.json`,
and per-view RGB, alpha, floating-point depth, visualized depth,
`rend_normal`, and `surf_normal` images. `rend_normal` is the alpha-composited
primitive normal; `surf_normal` is reconstructed from neighboring expected-depth
points and is the target used by the 2DGS normal-consistency loss.

## Evaluation

Evaluate the best checkpoint and calculate PSNR, SSIM, and LPIPS:

```bash
python source/evaluate.py \
    --scene_path /path/to/scene \
    --model_path output/<run-id> \
    --resolution 2
```

`evaluate.py` reads `checkpoint_metadata.json`, selects `best.checkpoint`, and
renders both the base and FlaRe models. Results are stored in:

```text
output/<run-id>/stats/evaluation.json
output/<run-id>/renders/evaluation_<iteration>/
```

The evaluation resolution must match the resolution used for training. The
first LPIPS evaluation may download pretrained VGG-16 weights through
`torchvision`.

## Geometry editing

Train a geometry-aware model, for example on the Mip-NeRF360 Garden scene:

```bash
python source/train.py \
    --source_path /path/to/mipnerf360/garden \
    --resolution 4 \
    --config garden_editing
```

FlaRe primitives can be exported as editable octagonal faces and fitted back
after an edit or animation in Blender. Export the trained model from the
repository root:

```bash
python source/export_edit_ply.py \
    --checkpoint output/<run-id>/checkpoints/best.checkpoint \
    --output edits/base.ply
```

After editing, put one or more PLY frames in a directory and render them from a
fixed test camera:

```bash
python source/render_edit_animation.py \
    --checkpoint output/<run-id>/checkpoints/best.checkpoint \
    --ply_dir edits/frames \
    --scene_path /path/to/scene \
    --output_dir edits/renders \
    --camera 0 \
    --resolution 2
```

The scene path, resolution, image directory and background color must match
training. The renderer sorts PLY filenames naturally and writes one numbered PNG
per frame. Use `--pattern 'frame_*.ply'` to select a sequence or `--mode base`
for the explicit RGB model.

The same renderer can apply a procedural sine displacement without modifying
the PLY in Blender:

```bash
python source/render_edit_animation.py ... \
    --deform sin \
    --deform_amplitude 0.1 \
    --deform_frequency 8.0 \
    --deform_rotation_z 30 \
    --phase_shift 90
```

`--phase_shift sweep` renders `base.ply` at phases 0 through 359. Add `--fade`
to prepend and append amplitude transitions; `--fade_frames 60` produces 480
frames in total. `sweep_camera` distributes the 360 sweep phases across all
test cameras. For example, a Garden animation can be rendered with:

```bash
python source/render_edit_animation.py \
    --checkpoint output/<run-id>/checkpoints/best.checkpoint \
    --ply_dir edits/frames \
    --scene_path /path/to/mipnerf360/garden \
    --output_dir edits/garden_sweep \
    --camera 0 \
    --resolution 4 \
    --deform sin \
    --deform_amplitude 0.5 \
    --deform_frequency 2.0 \
    --phase_shift sweep \
    --fade \
    --fade_frames 60
```

Sweep modes require `<ply_dir>/base.ply` and load that file only once.

Each face and its eight vertices represent one primitive. Do not add, remove,
merge, subdivide or reorder vertices in Blender. The exporter includes
`primitive_id` and `corner_id` attributes when the PLY importer/exporter keeps
custom properties; otherwise loading falls back to vertex or face order. Keep
geometry in the training scene's coordinate system by disabling axis conversion
on import and export (or applying exactly inverse conversions).

Run the CPU interchange smoke test with:

```bash
python scripts/smoke_test_edit_io.py
```

## Appearance-feature visualization

Render the first three principal components of the learned per-Gaussian
appearance embeddings through the scene:

```bash
python source/render_style.py \
    --source_path /path/to/scene \
    --checkpoint /path/to/best.checkpoint \
    --output_path feature-renders \
    --resolution 4 \
    --render_pca_embeddings
```

The PCA colors replace only the final RGB assigned to each primitive. FlaRe
opacity, geometry, visibility and front-to-back compositing are unchanged. The
command writes the rendered test views, `pca_embedding_colors.pt`, and a metrics
file marked `MODE: PCA embedding colors`; PSNR is intentionally not computed for
false-color output. The `.pt` file contains both the full pre-PCA embeddings
under `conditioning` and the projected RGB values under `colors`. Use
`--pca_color_percentile` to control outlier clipping.

A checkpoint-only diagnostic exports PCA scatter plots, the projection, basis,
variance ratios and a CSV table without invoking CUDA:

```bash
python source/visualize_appearance_embeddings.py \
    /path/to/best.checkpoint \
    --output_dir appearance-pca
```

## Appearance style transfer

The style-transfer driver uses CLIPGaussian-based optimization with the FlaRe
checkpoint format and differentiable renderer. It optimizes the per-primitive
descriptors, the shared appearance decoder, and geometry by default. The
view-direction component remains frozen unless `--style_train_view_branch` is
passed. Disable decoder or geometry updates with `--no_style_model_finetune` or
`--no_style_geometry_finetune`.

Image-guided stylization uses immutable renders of the loaded checkpoint as its
content and directional-CLIP references. For example, run a Starry
Night stylization of Garden with:

```bash
HF_HOME=/path/to/model-cache \
TORCH_HOME=/path/to/model-cache/torch \
python source/render_style.py \
    --source_path /path/to/garden \
    --checkpoint /path/to/iter_BEST.checkpoint \
    --output_path style-renders/starry_night \
    --images images_4 \
    --style_image /path/to/starry_night.jpg \
    --style_num_views 4
```

This configuration was tested with the Wikimedia/Google Art Project image of
Vincent van Gogh's *The Starry Night*.

Text guidance is available through `--style_prompt`. The first run downloads
OpenAI ViT-B/32 weights through `open_clip_torch` and ImageNet VGG-19 weights
through `torchvision`.

Optimization previews, `losses.txt`, the learned latent transform, and a
reloadable `styled_model.checkpoint` are written to
`<output_path>/style_optimization/`. Final styled test views and `metrics.txt`
are written directly to `<output_path>/`.

## Reproducing the paper benchmarks

The benchmark launchers train scenes sequentially with default optimization
parameters and then run the complete best-checkpoint evaluation.

### Mip-NeRF360

The launcher excludes `flowers` and `treehill`, uses resolution `4` for outdoor
scenes (`bicycle`, `garden`, `stump`), and resolution `2` for indoor scenes
(`room`, `counter`, `kitchen`, `bonsai`):

```bash
PYTHON_BIN=/path/to/flare/bin/python \
MIPNERF360_ROOT=/path/to/mipnerf360 \
./scripts/run_mipnerf360.sh
```

### Deep Blending

The `drjohnson` and `playroom` scenes are trained at resolution `1`:

```bash
PYTHON_BIN=/path/to/flare/bin/python \
DEEP_BLENDING_ROOT=/path/to/deep_blending \
./scripts/run_deep_blending.sh
```

### Tanks and Temples

The `truck` and `train` scenes are trained at resolution `1`:

```bash
PYTHON_BIN=/path/to/flare/bin/python \
TANDT_ROOT=/path/to/tandt \
./scripts/run_tandt.sh
```

Every run receives a numeric `output/<run-id>` directory. Its per-scene
evaluation summary is available in `stats/evaluation.json`. Do not launch multiple
training scripts concurrently in the same checkout because run IDs are
allocated from the shared `output/` directory.


## Additional scripts

Calculate the unpadded inference-model size and inspect checkpoint tensor
shapes:

```bash
python scripts/calculate_model_size.py \
    output/<run-id>/checkpoints/best.checkpoint
```

Run a quick CPU smoke test of the controlled LUT/hash benchmark:

```bash
python scripts/encoding_2d_benchmark.py \
    --device cpu \
    --steps 10 \
    --batch-size 256 \
    --eval-resolution 32 \
    --log-every 5 \
    --configs lut,hash_512 \
    --functions smooth \
    --seeds 0
```

See `python scripts/encoding_2d_benchmark.py --help` for the full paper
configuration and output controls.

## Citation

If you use FlaRe in your work, please cite:

```bibtex
@article{byrski2026flare,
  title={Floating Radiance Networks},
  author={Byrski, Krzysztof and Tobiasz, Rafal and Wilczynski, Grzegorz and Zielinski, Mikolaj and Baran, Dawid and Belter, Dominik and Tabor, Jacek and Spurek, Przemyslaw},
  journal={arXiv preprint arXiv:2608.05920},
  year={2026},
  url={https://arxiv.org/abs/2608.05920}
}
```

## License

FlaRe is distributed under the research-only [3D Gaussian Splatting License](LICENSE).
The repository includes code derived from the original 3D Gaussian Splatting
implementation; please review the license before use or redistribution.
