from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class Paths:
    repo_root: Path = Path(__file__).resolve().parents[1]
    data_dir: Path = repo_root / "data"
    raw_dir: Path = data_dir / "raw"
    db_dir: Path = data_dir / "db"
    mlruns_dir: Path = repo_root / "mlruns"

DEFAULT_PATHS = Paths()

