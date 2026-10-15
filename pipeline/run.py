"""One command to refresh Hatch's Picks:
sportsbook odds (The Odds API) -> download missing seasons + always the current season (nflverse)
-> features -> walk-forward model -> export -> website data in docs/data.

    python3 pipeline/run.py            # daily refresh
    python3 pipeline/run.py --all      # re-download every season (~500 MB)
"""
import subprocess
import sys
import urllib.request
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "raw"
REL = "https://github.com/nflverse/nflverse-data/releases/download"
SOURCES = {  # local prefix: (release path template, first season)
    "pbp": ("pbp/play_by_play_{y}.parquet", 1999),
    "inj": ("injuries/injuries_{y}.parquet", 2009),
    "snap": ("snap_counts/snap_counts_{y}.parquet", 2012),
}
GAMES = "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv"


def fetch(url, dest):
    tmp = dest.with_suffix(dest.suffix + ".part")
    urllib.request.urlretrieve(url, tmp)
    tmp.replace(dest)
    print(f"  {dest.name}: {dest.stat().st_size / 1e6:.1f} MB")


def download(refresh_all=False):
    RAW.mkdir(exist_ok=True)
    today = date.today()
    current = today.year if today.month >= 8 else today.year - 1
    fetch(GAMES, RAW / "games.csv")
    for key, (tpl, first) in SOURCES.items():
        for y in range(first, current + 1):
            dest = RAW / f"{key}_{y}.parquet"
            if refresh_all or y == current or not dest.exists():
                try:
                    fetch(f"{REL}/{tpl.format(y=y)}", dest)
                except Exception as e:  # current-season file may not exist yet in August
                    print(f"  skip {dest.name}: {e}")


def step(script):
    print(f"\n== {script}")
    subprocess.run([sys.executable, str(ROOT / "pipeline" / script)], check=True, cwd=ROOT)


if __name__ == "__main__":
    for d in ("raw", "data", "output", "docs/data"):
        (ROOT / d).mkdir(parents=True, exist_ok=True)
    print("== download")
    download("--all" in sys.argv)
    for s in ("odds.py", "features.py", "model.py", "export.py", "site_export.py"):
        step(s)
    print("\nDone -> docs/data")
