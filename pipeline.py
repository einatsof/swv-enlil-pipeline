"""
End-to-end WSA-Enlil pipeline: NOAA S3 → artifacts → Cloudflare R2.

Designed for GitHub Actions (cron + manual dispatch), runnable locally with
--dry-run (everything except upload/prune; needs no credentials).

Steps:
  1. Resolve the run SWPC itself publishes (its public animation names it) and
     find that prefix on s3://noaa-wsa-enlil-pds. **Not the newest prefix** —
     the bucket also carries single-CME analysis runs; see official_run_number().
  2. Idempotency: read enlil/latest.json straight from R2 (the source of truth
     the worker merely serves). Same run ⇒ exit 0. A credential-free --dry-run
     falls back to the public worker endpoint.
  3. Download pv-tim frames at FRAME_STRIDE (3 ⇒ 3-hourly, ~57 files ≈ 435 MB)
     plus metadata.json and evo.earth.nc, in parallel.
  4. extract.py → u8 artifacts (volumes off until the 3D page needs them).
  5. Upload to R2 enlil/<runId>/…, then write enlil/latest.json last — the
     worker serves latest.json, so a failed upload can't leave it pointing at
     a half-uploaded run.
  6. Prune R2 to the newest KEEP_RUNS run prefixes.

R2 credentials (upload/prune only), from environment / Actions secrets:
  R2_ACCOUNT_ID, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY
"""

import argparse
import concurrent.futures
import json
import os
import re
import sys
import tempfile
import urllib.request

import boto3
from botocore import UNSIGNED
from botocore.config import Config

from extract import extract_run

NOAA_BUCKET = 'noaa-wsa-enlil-pds'
R2_BUCKET = os.environ.get('R2_BUCKET', 'spaceweatherviz-textures')
WORKER_RUN_URL = 'https://spaceweatherviz-api.lakitzi.workers.dev/api/enlil/run'
# SWPC's public WSA-Enlil animation — the authority on which run is operational.
SWPC_ANIMATION_URL = 'https://services.swpc.noaa.gov/products/animations/enlil.json'
USER_AGENT = 'swv-enlil-pipeline (+https://github.com/einatsof/swv-enlil-pipeline)'
FRAME_STRIDE = 3
KEEP_RUNS = 10
DOWNLOAD_WORKERS = 8

RUN_PREFIX_RE = re.compile(r'^wsa_enlil\.(\d{8})(?:_(\d+))?/$')
# e.g. /images/animations/enlil/enlil_com2_58491_20260903T220000.jpg
ANIMATION_FRAME_RE = re.compile(r'enlil_\w+?_(\d+)_\d{8}T\d{6}\.jpg')


def noaa_client():
    return boto3.client('s3', config=Config(signature_version=UNSIGNED))


def r2_client():
    account = os.environ['R2_ACCOUNT_ID']
    return boto3.client(
        's3',
        endpoint_url=f'https://{account}.r2.cloudflarestorage.com',
        aws_access_key_id=os.environ['R2_ACCESS_KEY_ID'],
        aws_secret_access_key=os.environ['R2_SECRET_ACCESS_KEY'],
        region_name='auto',
    )


def list_run_prefixes(s3):
    """All top-level run prefixes, oldest → newest (names sort chronologically)."""
    prefixes, token = [], None
    while True:
        kw = dict(Bucket=NOAA_BUCKET, Delimiter='/', MaxKeys=1000)
        if token:
            kw['ContinuationToken'] = token
        r = s3.list_objects_v2(**kw)
        prefixes += [p['Prefix'] for p in r.get('CommonPrefixes', [])]
        if not r.get('IsTruncated'):
            break
        token = r['NextContinuationToken']
    return sorted(p for p in prefixes if RUN_PREFIX_RE.match(p))


def describe_run(s3, prefix):
    """Validate a run prefix and return its artifact keys, or None if unusable."""
    r = s3.list_objects_v2(Bucket=NOAA_BUCKET, Prefix=prefix, MaxKeys=1000)
    keys = [k['Key'] for k in r.get('Contents', [])]
    pvtims = sorted(k for k in keys if '/pv-ready-data-' in k and '/pv-tim.' in k)
    if len(pvtims) < 20:
        print(f'{prefix}: {len(pvtims)} pv frames, skipping')
        return None
    meta = [k for k in keys if k.endswith('/metadata.json') and '/pv-ready-data-' in k]
    evo = [k for k in keys if k.endswith('/evo.earth.nc') and '/pv-ready-data-' in k]
    if not meta or not evo:
        print(f'{prefix}: pv frames but missing metadata/evo, skipping')
        return None
    m = RUN_PREFIX_RE.match(prefix)
    run_id = m.group(1) + (f'_{m.group(2)}' if m.group(2) else '')
    return {'prefix': prefix, 'runId': run_id, 'pvtims': pvtims,
            'meta': meta[0], 'evo': evo[0]}


def official_run_number():
    """The run number SWPC is currently standing behind, or None.

    ⚠️ **The newest prefix in the bucket is NOT the operational forecast.** NOAA
    publishes every WSA-Enlil run it makes, and only some are the full-heliosphere
    forecast; the rest are single-CME analysis runs an analyst fires to size up one
    new event. They are indistinguishable from the inside — `metadata.json` carries
    the same `project`, `case` and `observatory` for both, and `cone2bc.in`'s
    `lproj` reads `..._test_cone` on *every* run including the operational ones, so
    that string means nothing. The only difference is the cone list.

    Taking the newest is therefore not "slightly fresher", it is **occasionally a
    different product**: on 2026-09-06 the newest prefix was 58494, a run modelling
    exactly ONE CME (lat 0, lon 70, 633 km/s), while the operational 58491 carried
    seven. Rendering 58494 would have silently dropped six real CMEs from the
    heliosphere view — far worse than being a few hours stale.

    SWPC names the operational run itself: the frames behind its public WSA-Enlil
    animation are `enlil_com2_<runNumber>_<frameTime>.jpg`, and it renders images
    for that run only (verified: 58491 → 200, while 58489/58490/58492/58494 all
    404, and those images were published at 23:36 — over an hour *after* 58492
    existed, so SWPC had it and passed on it). That JSON is the authority.

    Cost to accept: promotion lags run creation by ~1.3 h, and an analysis run
    that never gets promoted means the operational run can be many hours old.
    That is the correct trade — a complete older forecast beats a partial newer one.
    """
    try:
        req = urllib.request.Request(SWPC_ANIMATION_URL, headers={'User-Agent': USER_AGENT})
        with urllib.request.urlopen(req, timeout=30) as r:
            frames = json.load(r)
    except Exception as e:
        print(f'warning: could not read {SWPC_ANIMATION_URL}: {e}')
        return None
    nums = [m.group(1) for m in
            (ANIMATION_FRAME_RE.search(f.get('url', '')) for f in frames) if m]
    if not nums:
        print('warning: no run number in the SWPC animation manifest')
        return None
    # Mid-rollover the manifest can briefly mix two runs; the majority is the
    # one being served. Either would be a legitimately official run.
    return max(set(nums), key=nums.count)


def find_official_run(s3, number):
    """The prefix for a given SWPC run number. Searches every prefix, not just
    recent ones — the operational run sits behind newer analysis runs by design."""
    suffix = f'_{number}/'
    for prefix in reversed(list_run_prefixes(s3)):
        if prefix.endswith(suffix):
            return describe_run(s3, prefix)
    print(f'run {number} named by SWPC is not in the bucket yet')
    return None


def find_newest_pv_run(s3):
    """Newest run with pv-tim files, official or not — `--allow-unofficial` only.
    Walks back a few in case the newest prefix is still being written."""
    for prefix in reversed(list_run_prefixes(s3)[-6:]):
        run = describe_run(s3, prefix)
        if run:
            return run
    raise SystemExit('no usable pv-ready run found in the newest 6 prefixes')


def published_run_id(r2):
    """Which run is currently published, from R2 — the source of truth the
    worker only serves. Reading it here (rather than over HTTP) keeps the check
    authoritative and independent of the API's availability or edge caching."""
    try:
        obj = r2.get_object(Bucket=R2_BUCKET, Key='enlil/latest.json')
        return json.load(obj['Body']).get('runId')
    except r2.exceptions.NoSuchKey:
        return None
    except Exception as e:
        # Fail open: a read failure shouldn't stop data production. Worst case
        # is one redundant re-upload of a run we already have.
        print(f'warning: could not read enlil/latest.json from R2: {e}')
        return None


def published_run_id_via_worker():
    """Credential-free fallback for --dry-run.

    ⚠️ The custom User-Agent is required: Cloudflare answers the default
    `Python-urllib/x.y` agent with 403 before the Worker ever runs (verified —
    curl and any custom UA get 200 on the same URL). Don't remove it.
    """
    try:
        req = urllib.request.Request(WORKER_RUN_URL, headers={'User-Agent': USER_AGENT})
        with urllib.request.urlopen(req, timeout=30) as r:
            return (json.load(r).get('run') or {}).get('runId')
    except Exception as e:
        print(f'warning: could not read {WORKER_RUN_URL}: {e}')
        return None


def download_run(s3, run, dest):
    keys = run['pvtims'][::FRAME_STRIDE]
    if run['pvtims'][-1] not in keys:
        keys.append(run['pvtims'][-1])
    keys += [run['meta'], run['evo']]
    print(f'downloading {len(keys)} files from {run["prefix"]}')

    def get(key):
        local = os.path.join(dest, key.split('/')[-1])
        s3.download_file(NOAA_BUCKET, key, local)
        return local

    with concurrent.futures.ThreadPoolExecutor(DOWNLOAD_WORKERS) as ex:
        list(ex.map(get, keys))


def upload_artifacts(r2, out_dir, run_id):
    names = sorted(os.listdir(out_dir))
    print(f'uploading {len(names)} artifacts to r2://{R2_BUCKET}/enlil/{run_id}/')
    def put(name, key=None):
        ct = 'application/json' if name.endswith('.json') else 'application/octet-stream'
        r2.upload_file(os.path.join(out_dir, name), R2_BUCKET,
                       key or f'enlil/{run_id}/{name}',
                       ExtraArgs={'ContentType': ct})
    with concurrent.futures.ThreadPoolExecutor(DOWNLOAD_WORKERS) as ex:
        list(ex.map(put, [n for n in names if n != 'meta.json']))
    put('meta.json')
    # latest.json LAST: it is the pointer the worker serves, so everything it
    # references must already exist.
    put('meta.json', key='enlil/latest.json')


def prune_runs(r2):
    r = r2.list_objects_v2(Bucket=R2_BUCKET, Prefix='enlil/', Delimiter='/')
    runs = sorted(p['Prefix'] for p in r.get('CommonPrefixes', []))
    for prefix in runs[:-KEEP_RUNS] if len(runs) > KEEP_RUNS else []:
        print(f'pruning {prefix}')
        token = None
        while True:
            kw = dict(Bucket=R2_BUCKET, Prefix=prefix, MaxKeys=1000)
            if token:
                kw['ContinuationToken'] = token
            page = r2.list_objects_v2(**kw)
            objs = [{'Key': k['Key']} for k in page.get('Contents', [])]
            if objs:
                r2.delete_objects(Bucket=R2_BUCKET, Delete={'Objects': objs})
            if not page.get('IsTruncated'):
                break
            token = page['NextContinuationToken']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dry-run', action='store_true',
                    help='download + extract only; no R2 access, no credentials needed')
    ap.add_argument('--force', action='store_true',
                    help='process even if the worker already serves this run')
    ap.add_argument('--keep-out', default=None,
                    help='write artifacts here instead of a temp dir (implies inspectable output)')
    ap.add_argument('--allow-unofficial', action='store_true',
                    help='take the newest run in the bucket instead of the one SWPC '
                         'publishes — for local inspection only; the newest is often '
                         'a single-CME analysis run (see official_run_number)')
    args = ap.parse_args()

    s3 = noaa_client()
    if args.allow_unofficial:
        run = find_newest_pv_run(s3)
        print(f'newest NOAA run (UNOFFICIAL): {run["runId"]} ({len(run["pvtims"])} pv frames)')
    else:
        number = official_run_number()
        run = find_official_run(s3, number) if number else None
        if not run:
            # Deliberately not falling back to the newest prefix: publishing an
            # analysis run would replace the whole heliosphere with one CME.
            # Keeping the last good run is always the safer failure.
            print('could not resolve the operational run — leaving the published run in place')
            return
        print(f'SWPC operational run: {run["runId"]} ({len(run["pvtims"])} pv frames)')

    # Build the R2 client up front (not just before upload) so a bad/missing
    # credential fails in seconds rather than after a 435 MB download.
    have_creds = all(os.environ.get(k) for k in
                     ('R2_ACCOUNT_ID', 'R2_ACCESS_KEY_ID', 'R2_SECRET_ACCESS_KEY'))
    if not have_creds and not args.dry_run:
        raise SystemExit('R2_ACCOUNT_ID / R2_ACCESS_KEY_ID / R2_SECRET_ACCESS_KEY '
                         'must be set (or pass --dry-run)')
    r2 = r2_client() if have_creds else None

    current = published_run_id(r2) if r2 else published_run_id_via_worker()
    print(f'currently published: {current}')
    if current == run['runId'] and not args.force:
        print('up to date — nothing to do')
        return

    with tempfile.TemporaryDirectory(prefix='enlil_') as tmp:
        raw = os.path.join(tmp, 'raw')
        out = args.keep_out or os.path.join(tmp, 'out')
        os.makedirs(raw)
        download_run(s3, run, raw)
        extract_run(raw, out, run_id=run['runId'], volumes=False)

        if args.dry_run:
            print('dry run — skipping upload/prune')
            return
        upload_artifacts(r2, out, run['runId'])
        prune_runs(r2)
    print('done')


if __name__ == '__main__':
    main()
