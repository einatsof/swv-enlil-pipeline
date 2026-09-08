# swv-enlil-pipeline

Extracts compact web artifacts from NOAA WSA-Enlil model runs for
[SpaceWeatherViz](https://spaceweatherviz.com)'s monitor and heliosphere visualization.
Runs on GitHub Actions hourly (`.github/workflows/collect.yml`) and publishes to
Cloudflare R2, from which the site's worker serves `/api/enlil/run` and
`/api/enlil/frame/...`.

## Pipeline (`pipeline.py`)

1. Resolve the **operational** run — the number in SWPC's public animation manifest
   (`services.swpc.noaa.gov/products/animations/enlil.json`, frames named
   `enlil_com2_<run>_<time>.jpg`) — and find that prefix in the bucket.
   ⚠️ **Not the newest prefix.** NOAA publishes every run it makes, and many are
   single-CME analysis runs rather than the full forecast. They look identical from
   the inside: same `project`/`case`/`observatory` in `metadata.json`, and
   `cone2bc.in`'s `lproj` says `..._test_cone` on *every* run, operational ones
   included — that string is not a marker. Only the cone list differs, and taking
   the newest silently swaps the product: on 2026-09-06 the newest prefix (58494)
   modelled one CME while the operational run (58491) carried seven. SWPC renders
   animation frames for the operational run only (58491 → 200; 58489/58490/58492/58494
   → 404), which is what makes the manifest authoritative. Promotion lags creation
   by ~1.3 h; if the manifest can't be read or names a run not yet in the bucket the
   pipeline exits without republishing, keeping the last good run.
   `--allow-unofficial` restores newest-wins for local inspection.
2. Idempotency check against the **live worker** (`/api/enlil/run` is public, so this
   needs no credentials and behaves identically locally and in CI). Same run ⇒ exit.
3. Download the frames `frame_schedule()` selects (~113 files ≈ 860 MB), 8-way parallel.
   A run is a fixed 169 hourly frames where **frame N is `rundate` + (N − 48) h** —
   48 h of hindcast, then 120 h of forecast (verified on 20260907_58498:
   `rundate_cal` 2026-09-07T18, frame 0000 = 09-05T18:01Z, frame 0168 = 09-12T18:01Z;
   `extract.py` re-checks it per run and warns). Resolution goes where it is read:
   **frames 0000–0084 hourly, 0084–0168 3-hourly.**
   - *Hourly through rundate +36 h* because **every cone is injected in the
     hindcast** — across the four runs sampled every `cme_time` was ≤ `rundate`,
     spread over frames ~0–48, since cones come from observed events and are
     always in the past at run time — so the tracker follows their whole lives at
     full resolution. ⚠️ **It is not here to help cone attribution.** That was the
     original argument and it was backwards: a finer cadence reads the injection
     footprint *before* the DP cloud can be labelled at all. `ConeAttributor` now
     waits for the material (`ATTRIBUTION_WINDOW_HOURS`), so attribution is
     cadence-independent and this band neither helps nor hurts it.
     The band's forecast half is also the first ~31 h
     a run is on screen (it becomes displayable ~`rundate` +5–7 h: NOAA writes the
     pv data at +3–5 h, SWPC promotion adds ~1.3 h, then the hourly cron).
   - *3-hourly after* — the cadence the whole run used to get, so nothing regresses,
     and **tracking must continue through it**: mid and slow CMEs reach 1 AU inside
     this band (58491 lands four of seven cones at frame ≥ 84; 58498's single
     409 km/s cone crosses at frame ~138 and never leaves the 1.688 AU domain).
   - ⚠️ **No step may exceed 5 frames.** Spacing is not exactly 3600 s — it jitters
     (measured 3567–3634 s/frame on 20260903_58484), so a nominal 6 h stride reaches
     **6.06 h**, over `TrackingConfig.max_gap_hours` (6.0). That drops every link for
     the step: all tracks reborn with new IDs, and the cone attribution riding those
     links lost for the rest of the run.
   - A run that is not the standard 169-frame shape is sampled evenly at stride 3
     instead; index-based boundaries would land somewhere else entirely.
4. `extract.py`: quantize the fields and track DP blobs through the float 3D volumes
   with `regions.py`. Volume *exports* stay off until the site's 3D view lands.
5. Upload to `enlil/<runId>/…`; `enlil/latest.json` is written **last** so the pointer
   never references a half-uploaded run.
6. Prune R2 to the newest 10 runs.

Local dry run (no credentials): `python pipeline.py --dry-run [--force] [--keep-out out/x]
[--allow-unofficial]`.

Secrets (Actions → repository secrets): `R2_ACCOUNT_ID`, `R2_ACCESS_KEY_ID`,
`R2_SECRET_ACCESS_KEY` — an R2 API token scoped to object read/write on the one bucket.

## Data source

`s3://noaa-wsa-enlil-pds` (NOAA Open Data, anonymous). Since ~2024 each run prefix
(`wsa_enlil.YYYYMMDD_NNNNN/`) contains a `pv-ready-data-*/` directory with:

- `pv-tim.NNNN.nc` — hourly snapshots, pre-downsampled to **90 lon × 30 lat × 64 r**
  (~7.6 MB each, vs ~190 MB raw `tim.NNNN.nc`). Variables: `Density` (**already
  r²-scaled**: r²·N, ~flat ≈4–5 in quiet wind ⇒ ≈5 p/cc at 1 AU), `Vr` (km/s despite
  the `m/s` attr), `DP` (CME cloud tracer, 0 outside the CME), `T`, `Bx/By/Bz/Br`,
  `Pressure`. Grid: lon 2–358°E, lat ±58°, r 0.1125–1.688 AU, run frame is synodic.
- `metadata.json` — run dates + CME cone params (`cme_time`, `cme_latitude`,
  `cme_longitude`, `cme_cone_half_angle`, `cme_radial_velocity` as comma-joined strings).
- `evo.earth.nc` / `evo.l1.nc` / `evo.stereoa|b.nc` — dense time series at observers;
  `X/Y/Z` give the observer's position in the model frame.
- `cone2bc.in.*` — raw cone-model input (namelist).

Runs are published only when they contain CMEs — but **not every published run is the
operational forecast** (see step 1). Older prefixes (2023) hold a single ~31 GB
`full3D.tgz` instead — the pipeline only supports the pv layout.

Successive operational runs are often supersets: 58491 (7 cones, created 22:10) and
58492 (8 cones, 22:19) differ only by one appended cone and their `hash_digest`/`run_id`.
Do not read "more cones" as "more official" — 58492 was never promoted.

## Usage

```bash
python extract.py --run-dir <dir with pv-tim.*.nc + metadata.json + evo.earth.nc> \
                  --out out/<runId> [--no-volumes]
```

## Artifact format (consumed by the frontend)

Field binaries are u8; per-field encodings live in `meta.json.scales` and are **fixed**
(run-independent) so thresholds keep physical meaning:

- `ratio` — log2: `value = 2^(q/255 * (max-min) + min)` with min −2, max 4.5
  (0.25×–22.6× ambient; 1× ambient ⇒ q ≈ 78). Linear was rejected: shock noses reach
  20× while the quiet wind needs resolution around 1×.
- `vr` — linear: `value = min + q/255 * (max-min)`, 200–1200 km/s.
- `dp` — sqrt: `value = (q/255)² * max`, max 6 (resolution at faint cloud edges).

The density field stored is the **excess-density ratio** `Density / ambient(lat, r)`
where ambient is the azimuthal median of the first frame. This normalizes radial
contrast; it does not subtract structured background wind or identify CMEs.

| File | Shape (row-major) | Content |
|---|---|---|
| `slice_NNNN.bin` | 90 lon × 64 r | ecliptic-plane density ratio |
| `slvr_NNNN.bin` | 90 lon × 64 r | ecliptic-plane radial velocity (km/s) |
| `sldp_NNNN.bin` | 90 lon × 64 r | ecliptic-plane CME tracer |
| `vol_NNNN.bin` | 90 lon × 30 lat × 64 r | full ratio volume |
| `voldp_NNNN.bin` | 90 lon × 30 lat × 64 r | full DP volume |
| `line.bin` | nframes × 64 r × 3 | Sun→Earth line: ratio, vr, dp per frame |
| `meta.json` | — | frame times, grid, scales, ambient curve, Earth lon/lat, CME params |
| `blobs_<tag>.json` | — | float-DP 3D blob tracks, per-frame statistics/ancestry, and each region's `coneIdxs` |
| `labels_<tag>_NNNN.bin` | 90 lon × 64 r | **little-endian uint16** track IDs in the Earth-plane slice |

The "ecliptic" slice plane is the latitude cell nearest Earth's model latitude (from
`evo.earth.nc`), recorded as `grid.eclipticLatIndex`.

## Blob tracking (`regions.py`)

Every extraction now runs `BlobTracker` on the original floating-point `DP` and
`Vr` volumes, including with `--no-volumes`. Only the previous frame's cloud cells
and velocities are retained. The monitor currently still builds its outlines in
JavaScript; these additional artifacts are available through the Worker's existing
`/api/enlil/frame/<runId>/<file>` route for inspection and subsequent integration.

The tracker:

1. Labels six-connected cells with finite `DP > 0.25`, wrapping longitude only.
   Small clouds are retained by default (`min_cells=1`). A track can have multiple
   disconnected pieces in the displayed slice because connectivity is computed in 3D.
   NOAA pv latitude is stored **north to south**; this order is preserved. Longitude
   and radius increase. Uniform-axis checks tolerate source float32 rounding.
   NetCDF masked DP/Vr values become NaN, so finite fill-value sentinels cannot
   masquerade as clouds or enormous propagation speeds.
2. Predicts the next cloud mask using each cell's `Vr` and the actual time interval,
   in AU. Material leaving the radial domain is discarded, never clamped to its edge.
3. Links overlapping predicted/observed masks using intersection divided by the
   smaller component size (default minimum 0.25). When direct overlap fails, a one-cell
   tolerance in each grid direction is allowed; direct links take precedence. The
   tolerance score is capped at 1 and is **geometric evidence, not a probability**.
4. Keeps an ID for a one-to-one continuation. Merges, splits, and many-to-many
   reconfigurations receive new IDs with `parentTrackIds` and `originTrackIds`.
   Splitting after a merge retains the combined ancestry: shared DP cannot recover
   which original CME owns each fragment. No watershed boundary is invented.

When links branch, a component smaller than 5% of its largest sibling does not
cause a merge/split identity reset (`min_branch_fraction`). The small component
still receives its own track. This suppresses topology churn from one-cell tracer
specks while retaining young clouds; it is not an absolute detection-size cutoff.
Set the fraction to zero to preserve every qualifying lineage edge.

IDs are positive uint16 values, scoped to `(runId, artifactTag)`; zero means background.
A run makes a few hundred tracks at most (55 across 20260906_58495), so uint16 keeps the
per-frame label file at 11.5 KB rather than 23 KB; `update()` raises rather than wrapping
if a run ever exceeds it. Read the width from `meta.blobTracking.labelDtype`.
The tag hashes the algorithm version and configuration. **Bump `TRACKING_VERSION`
when changing the algorithm or schema** so immutable URLs never reuse stale labels.
Changing configuration generates a new tag automatically. To extract the currently
published run again, the existing pipeline `--force` option bypasses its run-ID check.

`meta.json.blobTracking` identifies the manifest, label pattern, byte order and layout.
The manifest includes grid metadata and a `frames` array in simulation-time order:

```json
{
  "frame": "0003",
  "time": "2026-09-01T03:00:00Z",
  "labelsFile": "labels_<tag>_0003.bin",
  "gapReset": false,
  "endedTrackIds": [],
  "regions": [{
    "trackId": 1,
    "event": "continue",
    "parentTrackIds": [],
    "originTrackIds": [1],
    "links": [{"trackId": 1, "overlap": 0.8, "method": "direct"}],
    "cellCount": 40,
    "sliceCellCount": 8,
    "centroid": {"lon": 180, "lat": 0, "rAU": 0.4},
    "radialRangeAU": [0.3, 0.5],
    "latitudeRangeDeg": [-4, 4],
    "peakDp": 1.2,
    "medianVrKms": 600,
    "coneIdxs": [0, 2]
  }]
}
```

`event` is `initial`, `birth`, `continue`, `merge`, `split`, `reconfigure`, or `gap`.
Parent/origin IDs describe track ancestry; `links` references the preceding frame.
`endedTrackIds` includes tracks lost to thresholding, exiting the domain, or replaced
at a topology change; it does not claim the material dissipated. `sliceCellCount=0`
means a cloud exists elsewhere in the volume but does not intersect this slice.
Centroids are spherical-cell-volume weighted, with a circular longitude mean (`null`
when no longitude direction is defined); coordinates remain in the source grid frame.

## Cone attribution (`extract.py: ConeAttributor`)

`regions.py` stays CME-agnostic; the join to NOAA's cone list lives in the extractor,
because it needs the **full 3D label volume**, and only the Earth-plane slice is ever
written. A consumer cannot redo it later.

A cone is read off its **injection footprint** — the label already present in the wedge
it occupies, from the inner boundary out to its ballistic nose at the first frame at or
after `cme_time`. It is not matched to a newly-born component, because a cone injected
into material that already exists never creates one: on run 20260906_58495 three of nine
cones (both at lat −4/lon −28, and lat +19/lon −75) produced **no birth event at all**.
The footprint read handles those and the ordinary case with one rule, and the widening
shell range also reaches cones injected before the run's first saved frame (spin-up).

⚠️ **The read waits for material rather than settling on one frame.** DP enters at the
inner boundary at `cme_time`, so a frame taken moments later cannot show the cloud — the
attributor defers until the nose has cleared one radial cell and keeps retrying until
`ATTRIBUTION_WINDOW_HOURS` (6 h) before recording that there is none. Without the wait,
attribution silently depended on the download cadence: hourly sampling moved the first
look at cone1 of `20260907_58497` from +2.08 h to **+5 min** (0 labelled cells in its
wedge at frame 0004, 8 at 0005, 12 at 0006) and untraced a real CME for the whole run.
The old 3-hourly cadence was not correct here, only lucky — a cone injected just before a
frame failed the same way. The window is bounded so a cone cannot latch onto material
that merely drifts through the wedge much later; a spin-up cone arrives already past it
and still resolves on frame 0.

Measured on that run, all nine cones resolved with **exactly one track in the footprint**
— nothing to tie-break — and across 19 frames every slice-visible region (≥5 cells) had
at least one cone, with **no cone ever appearing in two visible regions**.

Both directions of the join are published: regions carry `coneIdxs` (indices into
`meta.cmes`), and each cone carries `tracking`:

```json
{"frame": "0018", "trackIds": [48], "lastFrame": "0054",
 "footprintShare": 0.2424, "tracksInFootprint": 1}
```

`trackIds: []` with a `note` is an honest "no visible material", not a lookup failure.
`footprintShare` and `tracksInFootprint` are there to be audited — a low share, or more
than one track in a footprint, marks an attribution worth checking.

**A split assigns the cone to every fragment.** The tracker refuses to invent a boundary
inside connected material, so which fragment kept which CME is unrecoverable; naming one
would be a guess. **Consumers should key a pinned selection on `originTrackIds`, not
`trackId`** — `trackId` is reassigned at every merge/split, while the sorted origin set
is stable for the life of the physical cloud.

Filtering rule for display: track everything (`min_cells=1` keeps young clouds), then
drop specks at the consumer. Size is the weaker test — the reliable one is that every
genuine injection is born with `radialRangeAU[0] == grid.rad.min`, while tracer specks
are born mid-domain (0.94, 1.19 AU on the run above).

Limits: `regions.py` itself is geometric DP tracking only — cone attribution is a
separate stage (above) and DONKI matching happens in the Worker (`lib/cmeMatch.js`).
It does not find untagged CMEs in density. Missing detections end a track; there is no
hidden extrapolation through an empty frame. Gaps longer than six hours reset
association. Missing/negative velocity cells provide no prediction evidence. The pv
files have only radial velocity, so transverse motion is not reconstructed. A known
longitude drift can be supplied in grid degrees/day; its default is zero, not an
assumed solar rotation rate. Synthetic tests verify the mechanics; these thresholds
still need evaluation against multiple real runs before replacing browser attribution.
The saved run `20260903_58484` was also replayed through extraction (57 frames),
including verification that every slice label agrees with the manifest's counts.

CLI settings: `--dp-threshold 0.25` and `--tracking-longitude-rate 0.0`. Other settings
are exposed through `TrackingConfig` passed to `extract_run(..., tracking_config=...)`.
The existing field encodings and cone metadata are unaffected by these settings.

Run verification from this directory:

```bash
python -m unittest discover -s tests -v
```
