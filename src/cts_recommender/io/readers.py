from pathlib import Path
import json
import pandas as pd
import pyreadr

def _ensure_exists(path: Path):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"file not found: {p}")

def read_rdata(path: Path, key: str = 'mydata') -> pd.DataFrame:
    _ensure_exists(path)
    res = pyreadr.read_r(str(path))
    return res[key]

def read_json(path: Path):
    _ensure_exists(path)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
    
def read_parquet(path: Path) -> pd.DataFrame:
    _ensure_exists(path)
    return pd.read_parquet(path)

def read_csv(path: Path, **kwargs) -> pd.DataFrame:
    _ensure_exists(path)
    return pd.read_csv(path, encoding="utf-8", **kwargs)