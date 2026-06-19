"""
Experiment tracking — local run manifest (always on) + optional Weights & Biases.

Two layers, both safe to leave in place:

1. Run manifest  (zero-dependency, always on)
   Appends one row per experiment to models/run_manifest.csv with the exact
   provenance of every number: timestamp, git SHA, config name, a cheap data
   signature of the datacube, seed, and the headline metrics. This answers
   "which code + data produced this MAE?" without any external service.

2. Weights & Biases  (optional)
   Rich metric/config logging + cross-run comparison UI. It is OFF unless
   credentials exist, and it NEVER blocks a non-interactive run:
     - if a W&B API key is found (WANDB_API_KEY env or ~/.netrc) → mode "online"
     - else                                                      → mode "offline"
       (still logged locally under wandb/, you can `wandb sync` later)
     - HAZENET_WANDB=disabled turns it off entirely.
   Credentials are NEVER read from or written to the repo. Log in once on your
   machine with `wandb login`; the key lives in ~/.netrc, like ~/.cdsapirc.
"""
from __future__ import annotations

import os
import csv
import time
import hashlib
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST = os.path.join(ROOT, "models", "run_manifest.csv")


# ───────────────────────── provenance helpers ─────────────────────────
def git_sha(short: bool = True) -> str:
    try:
        args = ["git", "rev-parse", "--short" if short else "HEAD", "HEAD"]
        if not short:
            args = ["git", "rev-parse", "HEAD"]
        out = subprocess.check_output(args, cwd=ROOT, stderr=subprocess.DEVNULL)
        sha = out.decode().strip()
        # mark dirty tree so we never confuse uncommitted runs
        dirty = subprocess.call(["git", "diff", "--quiet"], cwd=ROOT,
                                stderr=subprocess.DEVNULL) != 0
        return sha + ("-dirty" if dirty else "")
    except Exception:
        return "nogit"


def data_signature(path: str) -> str:
    """
    Cheap content signature for a file OR a directory (e.g. datacube.zarr).
    Hashes the sorted (relpath, size, mtime) tuples — fast, and changes whenever
    the data is rebuilt. Not a cryptographic content hash, but enough to detect
    "did the inputs change between these two runs?".
    """
    if not path or not os.path.exists(path):
        return "missing"
    h = hashlib.sha1()
    if os.path.isfile(path):
        st = os.stat(path)
        h.update(f"{os.path.basename(path)}:{st.st_size}:{int(st.st_mtime)}".encode())
    else:
        for root, _, files in os.walk(path):
            for fn in sorted(files):
                fp = os.path.join(root, fn)
                try:
                    st = os.stat(fp)
                except OSError:
                    continue
                rel = os.path.relpath(fp, path)
                h.update(f"{rel}:{st.st_size}:{int(st.st_mtime)}".encode())
    return h.hexdigest()[:12]


def append_manifest(row: dict, path: str = MANIFEST) -> None:
    """Append one experiment row to the CSV manifest (writes header if new)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cols = ["timestamp", "git_sha", "project", "name", "data_sig", "seed",
            "mean_MAE", "worst_MAE", "seen_MAE", "new_MAE",
            "mae_2023", "bias_2023", "notes"]
    exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        if not exists:
            w.writeheader()
        row = {**row, "timestamp": row.get("timestamp", time.strftime("%Y-%m-%d %H:%M:%S"))}
        w.writerow(row)


# ───────────────────────── W&B mode resolution ─────────────────────────
def _wandb_mode() -> str:
    """Pick a W&B mode that never blocks a non-interactive run."""
    forced = os.environ.get("HAZENET_WANDB", "").lower()
    if forced in ("disabled", "offline", "online"):
        return forced
    if os.environ.get("WANDB_API_KEY"):
        return "online"
    # Ask wandb itself whether it has a usable key. This resolves from ALL the
    # places wandb stores it (env, ~/.netrc, %USERPROFILE%\.netrc on Windows,
    # ~/.config/wandb, keyring) — more robust than reading ~/.netrc by hand.
    try:
        import wandb
        if wandb.api.api_key:
            return "online"
    except Exception:
        pass
    return "offline"   # safe default: log locally, never prompt, sync later


# ───────────────────────── unified experiment handle ─────────────────────────
class Experiment:
    """
    One experiment run. Always writes the manifest on finish(); optionally
    streams to W&B. Degrades gracefully everywhere — tracking must never be the
    reason a training run fails.
    """
    def __init__(self, name: str, config: dict, project: str = "hazenet",
                 data_path: str | None = None, tags=None, notes: str = ""):
        self.name = name
        self.project = project
        self.config = dict(config or {})
        self.notes = notes
        self.seed = self.config.get("seed", "")
        self.data_sig = data_signature(data_path) if data_path else ""
        self.git = git_sha()
        self._wb = None

        mode = _wandb_mode()
        if mode != "disabled":
            try:
                import wandb
                self._wb = wandb.init(
                    project=project, name=name, config={**self.config,
                        "git_sha": self.git, "data_sig": self.data_sig},
                    tags=tags, notes=notes, mode=mode, reinit=True)
            except Exception as e:
                print(f"[tracking] W&B disabled ({e.__class__.__name__}: {e})")
                self._wb = None

    def log(self, metrics: dict, step: int | None = None) -> None:
        if self._wb is not None:
            try:
                self._wb.log(metrics, step=step)
            except Exception:
                pass

    def finish(self, summary: dict, status: str = "done") -> None:
        """Record final metrics to both the manifest and W&B."""
        if self._wb is not None:
            try:
                for k, v in summary.items():
                    self._wb.summary[k] = v
                self._wb.finish()
            except Exception:
                pass
        append_manifest({
            "git_sha": self.git, "project": self.project, "name": self.name,
            "data_sig": self.data_sig, "seed": self.seed,
            "mean_MAE": summary.get("mean_MAE"), "worst_MAE": summary.get("worst_MAE"),
            "seen_MAE": summary.get("seen_MAE"), "new_MAE": summary.get("new_MAE"),
            "mae_2023": summary.get("mae_2023"), "bias_2023": summary.get("bias_2023"),
            "notes": self.notes or status,
        })
