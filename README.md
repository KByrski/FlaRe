# FlaRe project page

Static, dependency-free project page for **FlaRe: Floating Radiance Networks**.

## Preview

Run a local server from this directory:

```bash
python3 -m http.server 8000
```

Then open `http://localhost:8000`.

## Add videos

The page has one main YouTube slot and six embedded MP4 loops. Their filenames and the YouTube embed snippet are listed in [`media/README.md`](media/README.md). To update a loop later, overwrite the matching MP4 and JPG poster—for example `media/garden_edit.mp4` and `media/garden_edit.jpg`. No HTML edit is needed.

Use H.264 MP4 for the widest browser support.

## Files

- `index.html` — content and media slots
- `styles.css` — responsive layout and visual system
- `script.js` — Gaussian wordmark, right-side section navigator, and BibTeX copy
- `favicon.svg` — Gaussian primitive favicon
- `media/` — put final videos here

The page is ready for GitHub Pages and needs no build step.
