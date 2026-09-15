# Virtual-server deploy (read-only)

This branch defaults to **read-only**. Writable register / Filter / Celov UI
requires an explicit `--writable` / `DASH_WRITABLE=1` and must not be used on
the lab VM.

## Safe entry points

| Command | Mode |
|---|---|
| `gunicorn src.GUI.wsgi:server` | **Forced** read-only + auth |
| `python -m src.GUI.app --secret-config …` | Read-only (branch default) |
| `python -m src.GUI.app --writable` | Desktop writes — local only |

## systemd

```bash
sudo cp deploy/dashboard-interactive.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now dashboard-interactive
sudo systemctl status dashboard-interactive
```

Confirm `ExecStart` uses `src.GUI.wsgi:server` and `DASH_READONLY=1`.

Default data folder: `/home/lab/data/Johannes/biofilm-microenvironments2`
(override with `DASH_DEFAULT_PROJECT`).

## nginx

Append `deploy/nginx-interactive.conf` to the dashboard SSL site, then reload nginx.

URL: `https://dashboard-drescher.biozentrum.unibas.ch/interactive/`
