#!/usr/bin/env python3
"""Add the `mup_base` identifying column (INTEGER NOT NULL DEFAULT 0) to
existing wallow registries so they stay readable after wallow.toml gained
the field (2026-09-22). Idempotent; touches `runs` and, if present,
`runs_superseded`. Existing rows are fixed-rate runs, so 0 is correct.
Usage: python scripts/add_mup_base_column.py <db> [<db> ...]"""
import sqlite3, sys
for path in sys.argv[1:]:
    c = sqlite3.connect(path)
    for table in ("runs", "runs_superseded"):
        if not c.execute("select name from sqlite_master where type='table' and name=?", (table,)).fetchone():
            continue
        cols = [r[1] for r in c.execute(f"pragma table_info({table})")]
        if "mup_base" in cols:
            print(f"{path}:{table}: already has mup_base"); continue
        c.execute(f"ALTER TABLE {table} ADD COLUMN mup_base INTEGER NOT NULL DEFAULT 0")
        c.commit(); print(f"{path}:{table}: added mup_base")
    c.close()
