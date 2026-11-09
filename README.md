# Photo to 3D (Phase 1 MVP)

> **Open the console:** https://adarshmalpeddiwar-beep.github.io/photo-to-3d/
> This is the browser version. It runs the whole pipeline in your browser, with no server and no sign-in.
> The Python server below is for running the app yourself.
> (If that link shows a 404, turn on GitHub Pages: Settings → Pages → Deploy from a branch → `main` / `/docs` → Save.)
>
> Claude artifact copy (sign-in required): https://claude.ai/artifact/X7qo1uz5vcx5juU7rEJZmb

Upload 1 to 5 photos of an object and get a print-ready STL and 3MF, with a silhouette
preview, a 3D viewer and print checks.

**Status: the pipeline, API, UI and print checks work end to end. The reconstruction step
is a placeholder.** The shape comes from extruding the photo outline into voxels. It is
not a real model of the object. Plug in a real backend (see "Adding a reconstruction
backend") before using the output for anything.

## Requirements

- Python 3.10 or newer
- NumPy and Pillow (see `requirements.txt`)
- A browser. The 3D viewer loads three.js from the jsDelivr CDN.

No web framework or task queue is needed. The server uses the Python standard library.

## Run it

```bash
pip install -r requirements.txt
python -m app.server --port 8000
# open http://127.0.0.1:8000
```

Run the tests:

```bash
pip install pytest
python -m pytest -q
```

### Configuration (environment variables)

| Variable | Default | Purpose |
| --- | --- | --- |
| `P3D_DATA_DIR` | `./data` | Where job folders (photos, masks, meshes, `job.json`) are stored. |
| `P3D_BACKEND` | `stub` | Reconstruction backend. Only `stub` is implemented. |
| `P3D_TOKEN` | unset | If set, every `/api` request needs `Authorization: Bearer <token>`. The browser asks for it once and remembers it. |

If you bind to anything other than localhost, set `P3D_TOKEN`. The server logs a warning
if you don't.

## How it works

Each job runs these stages in a background worker:

1. **segment**: find the object by comparing each pixel with the border colour (median),
   Otsu threshold, largest connected region.
2. **reconstruct**: the backend turns the silhouette into a mesh (stub: voxel extrusion).
3. **scale**: if you gave a reference width, scale so the mesh's width matches it. Otherwise
   normalise the longest side to 100 mm and mark the result as unscaled.
4. **clean**: weld duplicate vertices, drop degenerate triangles, remove unused vertices,
   make normals point outward, rest the model on Z = 0 and centre it.
5. **check**: print checks (below).
6. **export**: write `mesh.stl` (binary) and `model.3mf` (millimetres).

### Print checks

| Check | Pass / warn / fail rule |
| --- | --- |
| watertight | fail if any edge is open |
| manifold | fail if any edge is shared by more than two faces |
| build_volume | pass if it fits as placed (220 × 220 × 250 mm); warn if it only fits rotated; fail otherwise |
| overhangs | warn if more than 2% of surface area is steeper than 45° from vertical |
| face_count | warn above the 300k-triangle budget (decimation is not implemented) |
| wall_thickness | not checked in this build |

### Accuracy

- With a reference width, the **width** is calibrated. Depth and height are guessed from one photo.
- Without one, the size is a placeholder and the UI says so.
- The UI always shows the accuracy disclaimer.

## API

| Method and path | Purpose |
| --- | --- |
| `GET /api/jobs` | List jobs, newest first |
| `POST /api/jobs` | Create a draft. JSON body: `{"name": "...", "reference_width_mm": 60}` |
| `POST /api/jobs/{id}/photos` | Add a photo. Raw image bytes, JPEG, PNG or WebP, up to 5 per job |
| `POST /api/jobs/{id}/start` | Queue the job. Rate-limited (default 10 starts per 10 minutes) |
| `GET /api/jobs/{id}` | Status, stage, progress, scale info, print report, notes |
| `GET /api/jobs/{id}/files/{name}` | `mask.png`, `mesh.stl`, `model.3mf`, `photo-N.png` |

Uploads are re-encoded as PNG, which strips EXIF metadata including GPS tags.

## Adding a reconstruction backend

1. Subclass `ReconstructionBackend` in `app/reconstruct.py` and implement
   `reconstruct(mask) -> RawMesh`. Return vertices in backend units and set `unit`.
2. Register the name in `get_backend`.
3. Set `P3D_BACKEND` to that name. The rest of the pipeline stays the same.

Photogrammetry (multiple photos) and generative image-to-3D models both fit this interface.
Check each model's licence for commercial use first. See `docs/licences.md`.

## Limitations of this build

- The reconstruction is a placeholder (see above).
- Only one photo is used for the silhouette. Extra photos are stored but not used yet.
- No mesh decimation and no wall-thickness analysis.
- Jobs are stored on local disk and processed by one in-process worker thread. This is
  fine for low volume. For more volume, move the queue to Redis or Celery, and storage to S3.
- Single-user access control: one shared token, not accounts.
- Machinable (STEP) output is not implemented. It is Phase 2 in the brief.
- Segmentation assumes a plain background and even lighting.
- The frontend has been syntax-checked and its element IDs cross-checked against the HTML.
  The API has been tested end to end over HTTP. Viewer rendering has not been tested in a real browser.

## Project layout

```
app/
  config.py        settings from environment variables
  imaging.py       photo validation, EXIF-safe decoding, silhouette segmentation
  reconstruct.py   backend interface and the silhouette stub
  meshops.py       weld, cleanup, orientation, overhang and edge checks
  printcheck.py    print readiness report
  exporters.py     STL and 3MF read/write
  pipeline.py      the staged conversion
  jobs.py          job records, background worker, service layer
  server.py        HTTP API and static file server
web/               index.html, app.js, style.css (three.js viewer)
tests/             unit and pipeline tests on fixture meshes and synthetic photos
docs/licences.md   licences of the code and libraries used
```

## Hosted version

`hosted/index.html` is a self-contained page that runs the same pipeline entirely in the
browser (no server needed). It was published as a Claude artifact. Open it in any modern
browser, or host it as a static file. Files are processed locally and are never uploaded.
