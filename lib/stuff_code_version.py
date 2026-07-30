"""Version identity for the tracked Stuff+ analysis implementation."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
import subprocess


# Every source that can change collection, cohort construction, selection, or
# locked replay belongs to the analysis identity.  Keeping this list explicit
# makes additions to the locked surface a deliberate review decision.
ANALYSIS_SOURCE_FILES = (
    "build_fold_manifest.py",
    "final_evaluate.py",
    "search_xgboost_fast.py",
    "select_models.py",
    "stuff_mlb_temporal_final.py",
    "lib/build_outings.py",
    "lib/collect_fangraphs_stuff.py",
    "lib/collect_pitcher_statcast.py",
    "lib/evaluation.py",
    "lib/ordinal_linear.py",
    "lib/preprocessing.py",
    "lib/shared_tcn.py",
    "lib/stuff_cli.py",
    "lib/stuff_code_version.py",
    "lib/stuff_demo_data.py",
    "lib/stuff_experiment.py",
    "lib/stuff_final.py",
    "lib/stuff_mlb_dataset.py",
    "lib/stuff_selection.py",
    "lib/stuff_tabular.py",
    "lib/temporal_splits.py",
    "lib/xgboost_tpe.py",
    "requirements.txt",
)


def get_git_commit(repo_root: str | Path) -> str:
    """Return HEAD without importing the removed legacy ``lib.modeling``."""

    root = Path(repo_root).resolve()
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return "unknown"
    commit = result.stdout.strip()
    return commit if commit else "unknown"


def analysis_source_sha256(repo_root: str | Path | None = None) -> str:
    """Hash the normalized contents of every locked analysis source file."""

    root = (
        Path(repo_root).resolve()
        if repo_root is not None
        else Path(__file__).resolve().parents[1]
    )
    digest = sha256()
    for relative in ANALYSIS_SOURCE_FILES:
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"Analysis source file is missing: {path}")
        text = (
            path.read_text(encoding="utf-8")
            .replace("\r\n", "\n")
            .replace("\r", "\n")
        )
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(text.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def analysis_code_version(repo_root: str | Path | None = None) -> str:
    """Return Git identity plus a normalized analysis-source SHA-256."""

    root = (
        Path(repo_root).resolve()
        if repo_root is not None
        else Path(__file__).resolve().parents[1]
    )
    return f"{get_git_commit(root)}+analysis-sha256:{analysis_source_sha256(root)}"
