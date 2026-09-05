# One-time setup

Two things the code can't do for itself: publish the repo, and mint an R2 token.

## 1. R2 API token (Cloudflare dashboard)

**R2 → API → Manage API Tokens → Create API Token**

- Permission: **Object Read & Write**
- Scope: **specify bucket** → `spaceweatherviz-textures` (not "all buckets")
- TTL: forever

You get an **Access Key ID** and a **Secret Access Key** (shown once). The
**Account ID** is on the R2 overview page.

> This must be a **new** token, not the one hardcoded in
> `collector/r2-upload-helper.php` — that one is committed to git and is on the
> main repo's P0 list to rotate. Don't spread it further.

## 2. Public GitHub repo

Public is deliberate: Actions minutes are unlimited on public repos, which is
what makes the 3-hourly cron free. Nothing secret lives in the code — the
credentials above arrive only as Actions secrets.

```bash
# from the repo root
cd enlil-pipeline
git init -b main
git add .
git commit -m "WSA-Enlil → R2 pipeline"
gh repo create swv-enlil-pipeline --public --source=. --push
# (no gh? create the repo on github.com, then:)
#   git remote add origin git@github.com:<you>/swv-enlil-pipeline.git && git push -u origin main
```

`enlil-pipeline/` is currently tracked by the **main** SpaceWeatherViz repo. Pick one:

- **Standalone (recommended)** — `git rm -r --cached enlil-pipeline` in the main
  repo and add `enlil-pipeline/` to its `.gitignore`. One source of truth; the
  directory keeps working where it is.
- **Mirror** — leave it in both and publish with
  `git subtree push --prefix=enlil-pipeline <public-remote> main`. One checkout,
  but every publish is a subtree push.

Then add the three secrets: **Settings → Secrets and variables → Actions**
— `R2_ACCOUNT_ID`, `R2_ACCESS_KEY_ID`, `R2_SECRET_ACCESS_KEY`.

## 3. First run

**Actions → collect → Run workflow**, with `force` **checked** (the newest run is
already published from the manual upload, so an unforced run correctly exits as a
no-op and proves nothing).

Expect ~5–10 min: most of it downloading ~435 MB from S3. Then verify:

```bash
curl -s https://spaceweatherviz-api.lakitzi.workers.dev/api/enlil/run | head -c 200
```

`runId` should match the newest `wsa_enlil.*` prefix in the NOAA bucket, and
`generated` should be within minutes of the workflow run.

Run it a second time **without** `force` — it must exit in seconds with
"up to date — nothing to do". That is the idempotency check, and it is what keeps
the cron from re-downloading 435 MB every 3 hours.
