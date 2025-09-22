from pathlib import Path
import pandas as pd

def atomic_write_parquet(df: pd.DataFrame, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    df.to_parquet(tmp)           
    tmp.replace(out)             # atomic replace on same filesystem
