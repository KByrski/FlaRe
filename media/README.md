# Media slots

The page uses one main YouTube video and six short, muted MP4 loops.

## Main YouTube video

The page currently embeds `https://www.youtube.com/watch?v=r3bXuXjNqUM` using YouTube's privacy-enhanced domain. To replace it later, use:

```html
<iframe
  src="https://www.youtube-nocookie.com/embed/VIDEO_ID"
  title="FlaRe overview"
  allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
  allowfullscreen>
</iframe>
```

Replace `VIDEO_ID` with the final YouTube video ID. The `.youtube-frame` wrapper already supplies the correct 16:9 layout.

## Looping videos

Place the following H.264 MP4 files in this directory:

| File | Content |
| --- | --- |
| `garden_edit.mp4` | Geometry editing in the Garden scene |
| `garden_fiat126p.mp4` | Fiat 126p meshes added to Garden, showing reflections |
| `garden_style_transfer.mp4` | Garden orbit with multiple styles, including mosaic and Van Gogh, with ray tracing |
| `kitchen_style_transfer.mp4` | Kitchen style transfer with ray tracing |
| `drjohnson_materials.mp4` | Glass, metal, and mirror spheres in Dr. Johnson |
| `features_mesh.mp4` | Garden descriptor field, normal rendering, depth, and gray/colored extracted mesh |

The files are embedded as muted, autoplaying, looping videos. To update one later, overwrite the matching MP4 without changing its filename.

Matching `.jpg` poster frames are already included, so each slot has a useful image before its video starts playing.

Current web copies use H.264, `yuv420p`, `faststart`, and CRF 24–25 at 1280 px width. The longer multi-style Garden orbit uses 1152 px and CRF 26, reducing its 50 MB source to about 9.5 MB while retaining the fine mosaic and painterly texture.

The mesh-features sequence uses 1152×648, 24 FPS, and CRF 26. This reduces its 43 MB source to about 6.5 MB while preserving additional detail in the descriptor, normal, depth, and mesh stages.
