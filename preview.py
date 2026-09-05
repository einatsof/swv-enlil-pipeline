"""
Quick-look renderer for extracted artifacts: decodes u8 bins the same way the frontend
will and renders top-down polar heatmaps (+ optional animated GIF). Dev tool, not shipped.

Usage:
  python preview.py --artifacts out/<runId> [--frames 0000,0084] [--gif run.gif] [--size 720]
"""

import argparse
import json
import os

import numpy as np
from PIL import Image, ImageDraw

# site-ish palette: black -> deep blue -> cyan -> pale green -> white.
# Stop positions follow the log2 ratio encoding: 0.31 = 1x ambient, 0.46 = 2x,
# 0.62 = 4x, 0.77 = 8x, 1.0 = 22.6x.
RAMP_STOPS = [
    (0.00, (0, 0, 0)),
    (0.31, (10, 32, 72)),
    (0.46, (16, 110, 150)),
    (0.62, (60, 205, 190)),
    (0.77, (150, 240, 180)),
    (1.00, (255, 255, 255)),
]
DP_COLOR = np.array([255, 150, 60], dtype=float)  # warm CME-tracer overlay


def build_ramp():
    lut = np.zeros((256, 3), dtype=np.uint8)
    for i in range(256):
        x = i / 255.0
        for (x0, c0), (x1, c1) in zip(RAMP_STOPS, RAMP_STOPS[1:]):
            if x <= x1:
                t = (x - x0) / (x1 - x0) if x1 > x0 else 0.0
                lut[i] = [round(a + (b - a) * t) for a, b in zip(c0, c1)]
                break
    return lut


def render_frame(meta, slice_u8, dp_u8, size):
    g = meta['grid']
    n_lon, n_r = g['lon']['n'], g['rad']['n']
    r_max = g['rad']['max']
    sl = slice_u8.reshape(n_lon, n_r).astype(float)
    dp = dp_u8.reshape(n_lon, n_r).astype(float) / 255.0

    c = size / 2.0
    yy, xx = np.mgrid[0:size, 0:size]
    dx, dy = xx - c, c - yy                      # y up; 0 deg = +x, CCW
    r_au = np.hypot(dx, dy) / c * r_max
    phi = np.degrees(np.arctan2(dy, dx)) % 360.0

    lon0, lon1 = g['lon']['min'], g['lon']['max']
    ilon = np.clip(np.round((phi - lon0) / (lon1 - lon0) * (n_lon - 1)), 0, n_lon - 1).astype(int)
    r0, r1 = g['rad']['min'], g['rad']['max']
    irad = np.round((r_au - r0) / (r1 - r0) * (n_r - 1)).astype(int)
    inside = (irad >= 0) & (irad < n_r)
    irad = np.clip(irad, 0, n_r - 1)

    lut = build_ramp()
    img = lut[sl[ilon, irad].astype(np.uint8)].astype(float)
    dp_val = dp[ilon, irad] ** 2 * 6.0                       # decode sqrt encoding
    a = np.clip(dp_val * 0.5, 0, 0.85)[..., None]            # DP overlay
    img = img * (1 - a) + DP_COLOR * a
    img[~inside] = 8
    im = Image.fromarray(img.astype(np.uint8))

    d = ImageDraw.Draw(im)
    d.ellipse([c - 4, c - 4, c + 4, c + 4], fill=(255, 220, 120))   # Sun
    e_lon = np.radians(meta['earth']['lon'])
    er = 1.0 / r_max * c
    ex, ey = c + er * np.cos(e_lon), c - er * np.sin(e_lon)
    d.ellipse([ex - 4, ey - 4, ex + 4, ey + 4], outline=(120, 200, 255), width=2)
    d.ellipse([c - er, c - er, c + er, c + er], outline=(60, 60, 60))  # 1 AU ring
    return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--artifacts', required=True)
    ap.add_argument('--frames', default=None, help='comma list of frame ids (default: all)')
    ap.add_argument('--gif', default=None)
    ap.add_argument('--size', type=int, default=720)
    args = ap.parse_args()

    with open(os.path.join(args.artifacts, 'meta.json')) as f:
        meta = json.load(f)
    frames = args.frames.split(',') if args.frames else meta['frames']

    images = []
    for num in frames:
        sl = np.fromfile(os.path.join(args.artifacts, f'slice_{num}.bin'), dtype=np.uint8)
        dp = np.fromfile(os.path.join(args.artifacts, f'sldp_{num}.bin'), dtype=np.uint8)
        im = render_frame(meta, sl, dp, args.size)
        ImageDraw.Draw(im).text((10, 10), meta['times'][meta['frames'].index(num)],
                                fill=(200, 200, 200))
        images.append(im)
        if not args.gif:
            out = os.path.join(args.artifacts, f'preview_{num}.png')
            im.save(out)
            print('wrote', out)

    if args.gif and images:
        images[0].save(args.gif, save_all=True, append_images=images[1:],
                       duration=120, loop=0)
        print('wrote', args.gif, f'({os.path.getsize(args.gif)/1e6:.1f} MB)')


if __name__ == '__main__':
    main()
