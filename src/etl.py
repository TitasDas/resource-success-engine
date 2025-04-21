# Loads all relevant tables from the SQLite DB

import sqlite3
import pandas as pd
from typing import Dict, Union
from pathlib import Path
from src.config import DB_PATH


def load_tables(override_db_path: Union[str, Path, None] = None) -> Dict[str, pd.DataFrame]:
    """
    Loads required tables from SQLite database.

    Args:
        override_db_path: Optional path to a different .sqlite DB (e.g., synthetic)

    Returns:
        dict: Dictionary of DataFrames keyed by table names.
    """
    path = override_db_path if override_db_path else DB_PATH
    conn = sqlite3.connect(str(path))

    try:
        tables = {
            "Projects": pd.read_sql("SELECT * FROM Projects", conn),
            "People": pd.read_sql("SELECT * FROM People", conn),
            "Tasks": pd.read_sql("SELECT * FROM Tasks", conn),
            "AllocatedTasks": pd.read_sql("SELECT * FROM AllocatedTasks", conn),
            "LoggedHours": pd.read_sql("SELECT * FROM LoggedHours", conn),
            "ProjectTags": pd.read_sql("SELECT * FROM ProjectTags", conn),
            "PeopleTags": pd.read_sql("SELECT * FROM PeopleTags", conn),
            "Tags": pd.read_sql("SELECT * FROM Tags", conn),
        }
    finally:
        conn.close()

    return tables


if __name__ == "__main__":
    print("Loading tables from database...")
    tables = load_tables()
    for name, df in tables.items():
        print(f" Loaded {name:<15} — Rows: {df.shape[0]:>5} | Columns: {df.shape[1]}")
