# Orton Bradley storm fly-through

`render_linkedin.py` renders a separate social-video composition from the completed
72-hour FV1 event. It does not construct or advance a hydraulic solver. Conference
rendering defaults, the interactive application and existing source products are
unchanged. All generated media and images stay under the ignored `outputs/` tree.

Run from the project root with the pinned Python 3.14 environment:

```bash
FLOODWARP_LINKEDIN_ACTION=review .venv/bin/python -m examples.orton_bradley.render_linkedin
FLOODWARP_LINKEDIN_ACTION=preview .venv/bin/python -m examples.orton_bradley.render_linkedin
FLOODWARP_LINKEDIN_ACTION=full .venv/bin/python -m examples.orton_bradley.render_linkedin
```

Review renders six representative stills first. Preview renders the entire event
in eight seconds at 960 × 540. Full renders exactly 45 seconds at 1920 × 1080;
both use 30 fps, H.264, CRF 19, yuv420p, faststart and no audio. The existing
off-screen PyVista/VTK renderer uses software EGL, independent of the remote
desktop's WebGL capabilities and hydraulic CUDA jobs. No display server or new
imagery download is required.

Settings, paths, fonts, camera and timeline are explicit under the execution guard.
To resume, repeat the same command. Each profile retains its numbered PNG images
and identity record covering source hashes, selected hydraulic frames, renderer
hashes, font hashes, timeline, palette, dimensions, frame rate and camera settings.
An identity change is rejected before old images are reused; preserve the previous
profile directory and choose a new profile name for changed settings.

## Existing inputs

- Native event: `outputs/orton_bradley_visualisation/metadata/`,
  `diagnostics/` and all 865 five-minute instantaneous `frames/depth-*.npz` files.
- Prepared aerial colours: `outputs/orton_bradley_interactive/data/aerial.npz`,
  `aerial.json` and `manifest.json`. The existing renderer decodes these arrays.
- Original conference reference:
  `outputs/orton_bradley_visualisation/renders/conference/orton_bradley_fv1_3d.mp4`.

The local event records hydraulic source revision
`e480500facd0f33ece78c8b38a3a015cf7f43f7f`. The authoritative revision is written
to `source_audit.json` from `metadata/last_execution.json`, rather than inferred
from the newer rendering checkout.

All native frame hashes are checked before rendering and again afterwards. The
validation must already pass at exactly 259,200 seconds. Terrain/aerial grid and
south-to-north array orientation, full active-domain imagery coverage, the aerial
resource checksum, frame times, native FP64 depths and display-manifest identities
are checked. Missing or changed inputs stop execution.

## Composition and scientific representation

The gentle camera arc approaches the principal valley around peak flooding and
widens during recession. Terrain uses native 5 m geometry in EPSG:2193 / NZVD2016,
with the prepared Canterbury 2023 open-LINZ aerial colours. Water vertices are
exactly `DEM + h`, with vertical scale 1:1. A triangle is displayed only when all
three native vertices are active and have FP64 depth at least 0.01 m. This preserves
the existing display classification; very narrow wet strips can be omitted.

Water uses the established blue logarithmic depth palette, fixed at 0.01–7 m.
Opaque colour improves shallow-sheet visibility against the aerial imagery; it is
a scientific depth colour, not a photorealistic water material. There are no waves,
particles, flow synthesis, interpolated wet fronts or maximum-depth animation.
Aerial canopy/building colours are draped on the conditioned bare-earth terrain;
there is no added canopy/building geometry.

The production timeline is:

| Video time | Simulation time |
| --- | --- |
| 0–6 s | 0–18 h |
| 6–31 s | 18–30 h |
| 31–42 s | 30–72 h |
| 42–45 s | 72 h held |

Snapshots are held until the next selected native sample. The visible clock shows
the actual held snapshot, and a variable-speed note remains visible. Initial,
peak-discharge, peak-sampled-extent and final fields are explicitly included.
The local diagnostics put peak flow at 24:30 and peak sampled extent at 24:05,
within the requested slow section, so no control-point adjustment is needed.
Camera interpolation is independent of these held fields.

Opening/subtitle, compact depth legend, actual elapsed time, final credits and the
uncalibrated 100-year design-rainfall qualification are composited over the
landscape. Credits retain Patrick Durney / Lincoln Agritech Ltd, AGPL-3.0-only
software attribution, and the CC BY 4.0 terrain/imagery credits supplied locally.

## Outputs and checks

Everything is written beneath
`outputs/orton_bradley_visualisation/renders/linkedin/`:

- `orton_bradley_floodwarp_linkedin_45s_1080p.mp4`
- `orton_bradley_floodwarp_linkedin_preview.mp4`
- `orton_bradley_floodwarp_linkedin_cover.jpg` — actual peak-extent video frame
- `source_audit.json` and `render_report.json`
- `preview/` and `production/`: reusable numbered PNGs, identity, detailed report
  and contact sheet.

Encoding fully decodes every output frame and verifies frame count, dimensions
and frame rate, then checks duration, H.264/yuv420p, no audio and MP4 faststart.
Independent raw-scene probes compare dry/peak fields at one camera and a held
peak field at two cameras to catch static-scene bugs. Review the representative
images/contact sheets and the final encoded frames as well; automated checks
cannot certify visual quality. Reports retain source identities, exact schedules,
timings, sizes and SHA-256 values. No browser hosting or compression work is part
of this profile.

## Local production result (6 November 2026)

The completed film is 45.000 seconds, 1,350 frames, 1920 × 1080 at constant 30 fps,
28,272,268 bytes. The final software render/encode/verification pass took 288.2 s;
the 8-second preview took 26.0 s and is 2,444,209 bytes. This timing excludes the
source audit and the earlier camera review. The first camera pass remains archived
separately; a revised focal northing prevents a small coastal-edge crop.

All 865 native snapshots passed their hashes; the production schedule selects
655 distinct native fields. All 1,350 cameras keep the terrain in the viewport.
An independent decode also checked every presentation timestamp at exact 1/30 s
spacing, and the encoded start, rise, both peaks, recession and final frames were
visually reviewed. Additional records are `geometry_validation.json`,
`encoded_review_report.json` and `encoded_review/contact_sheet.png` in the output
directory. The existing validated dataset and conference/interactive products
remain intact; FV1 was not rerun and no imagery was downloaded.

Focused tests: 24 passed, one skipped. The full CPU-only suite (CUDA hidden to
avoid the separate hydraulic job) gave 1,093 passed, 645 skipped and one failure:
`test_enhanced_dispatch_preserves_full_positional_production_constructor` in
`tests/test_enhanced_production_api.py`. That failure reproduces independently in
the unchanged solver/test files; this profile does not change the constructor or
any numerical code. Logs are retained under the new output directory's `checks/`.
