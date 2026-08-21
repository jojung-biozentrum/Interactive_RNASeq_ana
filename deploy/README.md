# Virtual-server deploy

Read-only Dash viewer for the lab VM behind nginx, same pattern as
`~/www/dashboard/biofilm.py` (`server = app.server` + gunicorn).

Bronto (`/home/lab/data`) is mounted read-only. This branch therefore:

- never writes `project.yaml`
- has no dataset register / unregister
- has no native (tkinter) file dialogs
- has no Filter matrix tab
- has no Celov / CSV / other file exports

Plotly camera-download (PNG/SVG from the figure toolbar) stays: that is
browser-side, not a write on the server.

## 1. Prepare a project folder (writable disk)

Do **not** put `project.yaml` on Bronto if you still need to edit it.
Example:

```
/home/lab/www/dashboard/interactive-project/
  project.yaml
```

Point dataset paths at Bronto with **absolute** paths, e.g.
`/home/lab/data/.../countMatrix/....csv`.

`project.yaml` must already exist; the app will not create it.

## 2. Install

```bash
# clone or copy this branch onto the VM, e.g.
#   /home/lab/Interactive_RNASeq_ana

source /home/lab/anaconda3/etc/profile.d/conda.sh
conda activate plotly    # or a dedicated env
pip install -r requirements.txt
```

The existing `plotly` env may already have Dash/Plotly; still install
`dash-bootstrap-components`, `dash-auth`, `scikit-learn`, `umap-learn`, `gunicorn`.

## 3. systemd

Edit paths in `deploy/dashboard-interactive.service`, then:

```bash
sudo cp deploy/dashboard-interactive.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now dashboard-interactive
sudo systemctl status dashboard-interactive
```

Gunicorn imports `src.GUI.wsgi:server` (Flask/Werkzeug app on `app.server`).

Environment:

| Variable | Meaning |
|---|---|
| `DASH_URL_BASE_PATHNAME` | Must match the nginx location (`/interactive/`) |
| `DASH_DEFAULT_PROJECT` | Folder that already contains `project.yaml` |
| `DASH_SECRET_CONFIG` | TOML with `[auth] user` / `pwd`. Omit to run **without** a login. |

If `DASH_SECRET_CONFIG` is set, the file must exist or gunicorn will fail to start.
You can reuse `~/www/dashboard/.secret-rna-seq-viewer.toml` (same keys as the
gene viewers). An example is `deploy/secret-rna-seq-viewer.toml.example`.

## 4. nginx

Append `deploy/nginx-interactive.conf` to
`/etc/nginx/sites-available/dashboard-ssl` (lab user can edit that file),
then reload nginx.

URL: `https://dashboard-drescher.biozentrum.unibas.ch/interactive/`

With `DASH_SECRET_CONFIG` set, the browser shows HTTP Basic Auth (same prompt
as `/genes`). Leave the variable unset only for local testing.

## 5. Local check (without gunicorn)

```bash
python -m src.GUI.app --project /path/to/folder --url-base-pathname /interactive/
# optional login:
python -m src.GUI.app --project /path/to/folder --secret-config /path/to/.secret-rna-seq-viewer.toml
```
