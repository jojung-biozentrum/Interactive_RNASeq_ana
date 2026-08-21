"""Optional HTTP Basic Auth from a secret TOML (same idea as rna-seq-viewer)."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping


def _load_toml(path: Path) -> dict:
    try:
        import tomllib
    except ImportError:  # Python < 3.11
        import tomli as tomllib  # type: ignore[no-redef]

    with path.open("rb") as fh:
        return tomllib.load(fh) or {}


def load_basic_auth_users(path: str | Path) -> dict[str, str]:
    """Read ``[auth] user`` / ``pwd`` from a secret TOML (rna-seq-viewer schema)."""
    path = Path(path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Secret config not found: {path}")
    raw = _load_toml(path)
    auth = raw.get("auth") if isinstance(raw.get("auth"), dict) else {}
    users: dict[str, str] = {}

    nested = auth.get("users")
    if isinstance(nested, dict):
        users.update({str(k): str(v) for k, v in nested.items() if k and v is not None})

    user = auth.get("user") or auth.get("username")
    pwd = auth.get("pwd") if "pwd" in auth else auth.get("password")
    if user and pwd is not None and str(user).strip():
        users[str(user)] = str(pwd)

    if not users:
        raise ValueError(
            f"{path}: expected [auth] user and pwd (same as .secret-rna-seq-viewer.toml)."
        )
    return users


def enable_basic_auth(app, users: Mapping[str, str]) -> None:
    """Wrap a Dash app with dash_auth.BasicAuth."""
    import dash_auth

    dash_auth.BasicAuth(app, dict(users))
