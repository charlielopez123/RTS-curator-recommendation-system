from pathlib import Path
import json
import pandas as pd
import pyreadr

def read_rdata(path: Path, key: str = 'mydata') -> pd.DataFrame:
    res = pyreadr.read_r(str(path))
    return res[key]

def read_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)