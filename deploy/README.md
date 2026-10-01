# Virtual-server deploy (always read-only)

Goal: **one** process, always read-only, fixed data folder, basic auth.

No register / Filter / Celov / CSV export UI on this branch.

## 1. Code

```bash
cd /home/lab/Interactive_RNASeq_ana
git fetch && git checkout Virtual-server && git pull
```

Confirm:

```bash
grep DATA_ROOT src/GUI/runtime.py   # lab biofilm-microenvironments2 path
grep create_app src/GUI/wsgi.py
```

## 2. Secret TOML

`[auth] user` / `pwd` — e.g. `/home/lab/www/dashboard/secret-rna-seq-ana.toml`
(or reuse `.secret-rna-seq-viewer.toml`; set `DASH_SECRET_CONFIG` to match).

## 3. Install systemd (recommended)

```bash
bash deploy/install-dashboard-interactive.sh
```

Or manually:

```bash
sudo cp deploy/dashboard-interactive.service /etc/systemd/system/
# edit paths if your clone / conda / secret differ
sudo systemctl daemon-reload
sudo systemctl enable --now dashboard-interactive
sudo systemctl status dashboard-interactive
```

`ExecStart` **must** be `… gunicorn … src.GUI.wsgi:server -b 127.0.0.1:8052`.

## 4. nginx

Append `deploy/nginx-interactive.conf` next to `/genes` in
`/etc/nginx/sites-available/dashboard-ssl`, then:

```bash
sudo nginx -t && sudo systemctl reload nginx
```

URL: `https://dashboard-drescher.biozentrum.unibas.ch/interactive/`

## 5. Custom wrappers (if you already have one)

```python
import sys
sys.path.insert(0, "/home/lab/Interactive_RNASeq_ana")
from src.GUI.app import create_app

app = create_app(
    url_base_pathname="/interactive/",
    secret_config="/home/lab/www/dashboard/secret-rna-seq-ana.toml",
    default_project="/home/lab/data/Johannes/biofilm-microenvironments2",
)
server = app.server
```

Better: do not maintain a wrapper — use `src.GUI.wsgi:server` or
`deploy.interactive_wsgi:server`.

## 6. After every pull

```bash
sudo systemctl restart dashboard-interactive
# then Ctrl+F5 in the browser
```

## 7. Sanity check

UI: **no** “Working folder”, **no** Register / Unregister, **no** Filter / Celov / Export.

```bash
systemctl cat dashboard-interactive | grep -E 'ExecStart|DASH_'
ss -lptn | grep 8052
```

## Safe entry points

| Command | Mode |
|---|---|
| `gunicorn src.GUI.wsgi:server` | Always read-only |
| `create_app(…)` | Always read-only |
| `python -m src.GUI.app --secret-config …` | Always read-only |

## Security

### Virtual server

1. **Basic Auth TOML** — plaintext `user` / `pwd`. Keep the file mode `600`, lab-only password; nginx must proxy to gunicorn on `127.0.0.1` only.
2. **Never bind gunicorn publicly** — auth is app-level; rely on nginx TLS + localhost bind (`-b 127.0.0.1:8052`).
3. **Volcano filter DSL** uses a restricted `eval` in `volcano_condition.py` — authenticated users only; treat a public bind as unsafe. AST rewrite is a follow-up.
4. **Dataset paths** in `project.yaml` are sandboxed under the project root (`Project.resolve`); absolute / `..` escapes are rejected.
5. **Plotly client SVG/PNG download** is browser-side (expected; no server write).

### Publishing this repo

1. Keep personal WSL paths out of `src/`; institutional lab paths belong in `deploy/` only.
2. Never commit secret TOML (gitignored).
3. This branch is the server viewer — do not advertise it as a general writable desktop app (use a separate writable branch/`main` for that).
4. Internal hostname in this README is intentional for the lab; redact if the repo goes fully public outside the org.
