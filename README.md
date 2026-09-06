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
3. Download frames at stride 3 (3-hourly, ~57 files ≈ 435 MB), 8-way parallel.
4. `extract.py` (volumes off until the site's 3D view lands).
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

All binaries are u8; per-field encodings live in `meta.json.scales` and are **fixed**
(run-independent) so thresholds keep physical meaning:

- `ratio` — log2: `value = 2^(q/255 * (max-min) + min)` with min −2, max 4.5
  (0.25×–22.6× ambient; 1× ambient ⇒ q ≈ 78). Linear was rejected: shock noses reach
  20× while the quiet wind needs resolution around 1×.
- `vr` — linear: `value = min + q/255 * (max-min)`, 200–1200 km/s.
- `dp` — sqrt: `value = (q/255)² * max`, max 6 (resolution at faint cloud edges).

The density field stored is the **excess-density ratio** `Density / ambient(lat, r)`
where ambient is the azimuthal median of the first (pre-CME) frame — the CME is the
only bright object; 1.0 ≈ quiet wind.

| File | Shape (row-major) | Content |
|---|---|---|
| `slice_NNNN.bin` | 90 lon × 64 r | ecliptic-plane density ratio |
| `slvr_NNNN.bin` | 90 lon × 64 r | ecliptic-plane radial velocity (km/s) |
| `sldp_NNNN.bin` | 90 lon × 64 r | ecliptic-plane CME tracer |
| `vol_NNNN.bin` | 90 lon × 30 lat × 64 r | full ratio volume |
| `voldp_NNNN.bin` | 90 lon × 30 lat × 64 r | full DP volume |
| `line.bin` | nframes × 64 r × 3 | Sun→Earth line: ratio, vr, dp per frame |
| `meta.json` | — | frame times, grid, scales, ambient curve, Earth lon/lat, CME params |

The "ecliptic" slice plane is the latitude cell nearest Earth's model latitude (from
`evo.earth.nc`), recorded as `grid.eclipticLatIndex`.
