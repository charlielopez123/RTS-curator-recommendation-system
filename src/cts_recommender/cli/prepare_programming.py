import argparse
from pathlib import Path
from cts_recommender.pipelines.programming_pipeline import run_original_Rdata_programming_pipeline

def main():
    p = argparse.ArgumentParser(description="Prepare programming dataset")
    p.add_argument("--rdata", help="Path to .RData file")
    p.add_argument("--out", required=True, help="Path to output parquet")

    a = p.parse_args()
    _, out = run_original_Rdata_programming_pipeline(raw_file=Path(a.rdata), out_file=Path(a.out))
    print(f"Wrote {out}")

if __name__ == "__main__":
    main()