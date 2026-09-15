# Virtual-server deploy (read-only)

Goal: **one** process, **forced** read-only, fixed data folder, basic auth.

Writable register / Filter / Celov must not run on the lab VM.

## 1. Code

```bash
cd /home/lab/Interactive_RNASeq_ana
git fetch && git checkout Virtual-server && git pull
```

Confirm:

```bash
grep DEPLOYMENT_READONLY_DEFAULT src/GUI/runtime.py   # True
grep 'readonly=True' src/GUI/wsgi.py
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

This is **enough** and stays read-only (URL prefix forces readonly):

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

Read-only UI: **no** “Working folder”, **no** Register / Unregister, **no** Filter tab.

```bash
systemctl cat dashboard-interactive | grep -E 'ExecStart|DASH_'
ss -lptn | grep 8052
```

## Safe entry points

| Command | Mode |
|---|---|
| `gunicorn src.GUI.wsgi:server` | Forced read-only |
| `create_app(url_base_pathname="/interactive/", …)` | Forced read-only |
| `python -m src.GUI.app --writable` | Desktop only — do not use on VM |
