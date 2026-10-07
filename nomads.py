"""
NCEP NOMADS access for the daily ambient WSA-Enlil run.

    https://nomads.ncep.noaa.gov/pub/data/nccf/com/wsa_enlil/prod/wsa_enlil.<YYYYMMDD>/
        wsa_enlil.mrid00000000.suball.nc        ~131 MB  (three 2D cut planes)
        wsa_enlil.mrid00000000.inputs.tar.gz    ~0.7 MB  (enlil.in, grd.nc, bnd.nc, ...)

Constraints this module is shaped around:
  - **~2-day retention.** Only today's and yesterday's day folders exist. A run
    covers -48 h ... +120 h, so publishing the newest one each day is enough,
    and the hourly cron gets ~40 chances at each before it disappears.
  - **Files can be listed while still being written.** Anything modified less
    than SETTLE_MINUTES ago is left for the next tick.
  - **`mrid00000000` is a convention, not a guarantee.** It is the daily
    background run today (`ncmes=0`, NOAA's own job path says
    `wsa_enlil_bkgrnd`). If NCEP ever reuses the name for a CME run, publishing
    it as "ambient" would be wrong — so `enlil.in` is read and anything with
    `ncmes != 0` is refused.
  - **Polite access.** One HEAD per day folder per tick, one ~131 MB GET per
    new run, a descriptive User-Agent, and backoff on failure.
"""

import email.utils
import io
import os
import re
import tarfile
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone

NOMADS_BASE = 'https://nomads.ncep.noaa.gov/pub/data/nccf/com/wsa_enlil/prod/'
AMBIENT_SUBALL = 'wsa_enlil.mrid00000000.suball.nc'
AMBIENT_INPUTS = 'wsa_enlil.mrid00000000.inputs.tar.gz'
SETTLE_MINUTES = 15
DAY_DIR_RE = re.compile(r'href="(wsa_enlil\.(\d{8})/)"')
NCMES_RE = re.compile(r'\bncmes\s*=\s*(\d+)', re.IGNORECASE)
RETRIES = 3


def _request(url, method='GET', user_agent='swv-enlil-pipeline'):
    return urllib.request.Request(url, method=method, headers={'User-Agent': user_agent})


def _with_retries(fn, what):
    for attempt in range(RETRIES):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001 — network errors come in many shapes
            if attempt == RETRIES - 1:
                raise
            wait = 10 * 3 ** attempt
            print(f'{what} failed ({e}); retrying in {wait}s')
            time.sleep(wait)


def day_dirs(listing_html):
    """Day folders in a `prod/` index page, newest first."""
    return sorted({m.group(1) for m in DAY_DIR_RE.finditer(listing_html)}, reverse=True)


def parse_ncmes(enlil_in_text):
    """The `ncmes` namelist value from an `enlil.in`, or None if absent."""
    m = NCMES_RE.search(enlil_in_text)
    return int(m.group(1)) if m else None


def is_settled(last_modified, now=None, minutes=SETTLE_MINUTES):
    """True once a file has been untouched long enough to be complete."""
    now = now or datetime.now(timezone.utc)
    return (now - last_modified).total_seconds() >= minutes * 60


def head(url, user_agent):
    """(Last-Modified as aware datetime, Content-Length) for a URL, or None on 404."""
    def go():
        try:
            with urllib.request.urlopen(_request(url, 'HEAD', user_agent), timeout=60) as r:
                lm = email.utils.parsedate_to_datetime(r.headers['Last-Modified'])
                return lm.astimezone(timezone.utc), int(r.headers['Content-Length'])
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            raise
    return _with_retries(go, f'HEAD {url}')


def find_latest_ambient(user_agent):
    """The newest ambient run on NOMADS, or None.

    Returns {dir, suballUrl, inputsUrl, lastModified (ISO), size, settled}.
    Walks day folders newest-first, so a missing or not-yet-written run today
    falls back to yesterday's — which still covers now (-48 h ... +120 h).
    """
    def listing():
        with urllib.request.urlopen(_request(NOMADS_BASE, user_agent=user_agent), timeout=60) as r:
            return r.read().decode('utf-8', 'replace')
    for d in day_dirs(_with_retries(listing, f'list {NOMADS_BASE}')):
        url = NOMADS_BASE + d + AMBIENT_SUBALL
        info = head(url, user_agent)
        if not info:
            print(f'{d}: no ambient run')
            continue
        last_modified, size = info
        return {
            'dir': d.rstrip('/'),
            'suballUrl': url,
            'inputsUrl': NOMADS_BASE + d + AMBIENT_INPUTS,
            'lastModified': last_modified.strftime('%Y-%m-%dT%H:%M:%SZ'),
            'size': size,
            'settled': is_settled(last_modified),
        }
    return None


def download(url, dest, expected_size, user_agent):
    """Stream `url` to `dest`; the byte count must match Content-Length."""
    def go():
        with urllib.request.urlopen(_request(url, user_agent=user_agent), timeout=300) as r, \
                open(dest, 'wb') as f:
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
        got = os.path.getsize(dest)
        if expected_size is not None and got != expected_size:
            raise IOError(f'truncated download: {got} of {expected_size} bytes')
    _with_retries(go, f'GET {url}')


def ncmes_from_inputs(tar_bytes):
    """`ncmes` from the `enlil.in*` member of an inputs.tar.gz, or None."""
    with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode='r:gz') as tar:
        for member in tar.getmembers():
            name = member.name.rsplit('/', 1)[-1]
            if member.isfile() and name.startswith('enlil.in'):
                return parse_ncmes(tar.extractfile(member).read().decode('utf-8', 'replace'))
    return None


def fetch_ncmes(inputs_url, user_agent):
    def go():
        with urllib.request.urlopen(_request(inputs_url, user_agent=user_agent), timeout=120) as r:
            return r.read()
    return ncmes_from_inputs(_with_retries(go, f'GET {inputs_url}'))
