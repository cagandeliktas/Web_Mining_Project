"""One-time load: push the raw Sephora review CSVs into Snowflake.

Loads all five reviews_*.csv files into WEB_MINING.RAW.REVIEWS_RAW as a
single table. Columns are kept as-is (uppercased) with no type coercion -
this is the "raw" landing layer; cleanup and typing happens later in dbt
staging models, not here.

Needs SNOWFLAKE_ACCOUNT, SNOWFLAKE_USER, SNOWFLAKE_PASSWORD set as
environment variables (never pass credentials as command-line arguments -
they'd end up in your shell history).

Usage:
    export SNOWFLAKE_ACCOUNT=...
    export SNOWFLAKE_USER=...
    export SNOWFLAKE_PASSWORD=...
    python scripts/load_reviews_to_snowflake.py --data-dir /path/to/sephora_datasets
"""

import argparse
import os

import pandas as pd
import snowflake.connector
from snowflake.connector.pandas_tools import write_pandas

REVIEW_FILES = [
    "reviews_0-250.csv",
    "reviews_250-500.csv",
    "reviews_500-750.csv",
    "reviews_750-1250.csv",
    "reviews_1250-end.csv",
]

TABLE_NAME = "REVIEWS_RAW"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--database", default="WEB_MINING")
    parser.add_argument("--schema", default="RAW")
    parser.add_argument("--warehouse", default="COMPUTE_WH")
    args = parser.parse_args()

    conn = snowflake.connector.connect(
        account=os.environ["SNOWFLAKE_ACCOUNT"],
        user=os.environ["SNOWFLAKE_USER"],
        password=os.environ["SNOWFLAKE_PASSWORD"],
        warehouse=args.warehouse,
        database=args.database,
        schema=args.schema,
    )

    try:
        for i, filename in enumerate(REVIEW_FILES):
            path = os.path.join(args.data_dir, filename)
            print(f"Reading {filename} ...")
            df = pd.read_csv(path, low_memory=False)
            df.columns = [c.upper() for c in df.columns]

            print(f"Loading {len(df)} rows from {filename} into {TABLE_NAME} ...")
            success, _, nrows, _ = write_pandas(
                conn,
                df,
                TABLE_NAME,
                auto_create_table=(i == 0),
                overwrite=(i == 0),
            )
            print(f"  -> {filename}: success={success}, rows={nrows}")
    finally:
        conn.close()

    print("Done.")


if __name__ == "__main__":
    main()
