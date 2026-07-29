"""Consolidate run registries from multiple machines into one canonical store.

Historical registries were written by different wallow vintages and live in
per-machine SQLite files. Two operations are provided:

- ``convert``: copy a legacy registry into a fresh store created by the
  installed wallow (legacy files name the per-row id column ``run_uuid``;
  current wallow names it ``uuid``), rewriting artefact-path annotations to
  the canonical repo-anchored layout. The result is readable by every
  ``gc-*`` tool via ``--db``.

- ``merge``: combine N sources (priority = argument order) into one canonical
  registry. Rows are deduplicated on the identifying tuple
  (``identifying.IDENTIFYING_FIELDS``); for each tuple the winner is chosen by
  (completed status, artefact presence, source priority, latest completion).
  Losing and non-completed rows are preserved verbatim in a
  ``runs_superseded`` side table (wallow ignores extra tables) so every
  source remains reproducible from the canonical file. Winning rows' artefact
  dirs are copied under the canonical data root when they live elsewhere.

Sources are opened read-only and never modified.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

from .identifying import IDENTIFYING_FIELDS

#: SQL name of the per-row id column in legacy registries vs current wallow.
LEGACY_UUID_COL = "run_uuid"
UUID_COL = "uuid"


# ---------------------------------------------------------------------------
# helpers


def _connect_ro(db_path: str | Path) -> sqlite3.Connection:
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    return con


def _columns(con: sqlite3.Connection, table: str = "runs") -> list[str]:
    return [r[1] for r in con.execute(f"PRAGMA table_info({table})")]


def _read_rows(db_path: str | Path) -> tuple[list[dict], list[str]]:
    """All rows of a source registry as dicts keyed by canonical column names."""
    con = _connect_ro(db_path)
    try:
        cols = _columns(con)
        rows = [dict(r) for r in con.execute("SELECT * FROM runs")]
    finally:
        con.close()
    for row in rows:
        row.pop("id", None)  # autoincrement PK: never carried across stores
        if LEGACY_UUID_COL in row:
            row[UUID_COL] = row.pop(LEGACY_UUID_COL)
    cols = [UUID_COL if c == LEGACY_UUID_COL else c for c in cols if c != "id"]
    return rows, cols


def _canonical_paths(row: dict, data_root: Path) -> None:
    """Point a row's artefact annotations at the canonical layout in-place."""
    d = data_root / str(row["experiment_type"]) / str(row[UUID_COL])
    row["artefacts_dir"] = str(d)
    row["npz_path"] = str(d / "trace.npz")


def _create_store_db(dest: Path) -> None:
    """Create an empty registry with the installed wallow's schema."""
    from .store import get_schema

    os.environ.setdefault("WALLOW_JOURNAL_MODE", "DELETE")
    from wallow import Store

    Store(str(dest), schema=get_schema(), check_schema=False)


def _insert(con: sqlite3.Connection, table: str, rows: list[dict], cols: list[str]) -> None:
    sql = f"INSERT INTO {table} ({', '.join(cols)}) VALUES ({', '.join('?' for _ in cols)})"
    con.executemany(sql, [[row.get(c) for c in cols] for row in rows])


def _identifying_key(row: dict) -> tuple:
    return tuple(row[f] for f in IDENTIFYING_FIELDS)


def _npz_exists(row: dict, artefact_root: Path) -> bool:
    return (artefact_root / str(row["experiment_type"]) / str(row[UUID_COL]) / "trace.npz").is_file()


# ---------------------------------------------------------------------------
# convert


def convert(source_db: Path, dest_db: Path, data_root: Path) -> int:
    """Copy a legacy registry into a fresh current-wallow store."""
    rows, cols = _read_rows(source_db)
    for row in rows:
        _canonical_paths(row, data_root)
    if dest_db.exists():
        raise FileExistsError(dest_db)
    _create_store_db(dest_db)
    con = sqlite3.connect(dest_db)
    try:
        with con:
            _insert(con, "runs", rows, cols)
    finally:
        con.close()
    return len(rows)


# ---------------------------------------------------------------------------
# merge


@dataclass
class _Candidate:
    row: dict
    source_index: int
    source_db: str
    artefact_root: Path
    has_npz: bool = False

    def rank(self) -> tuple:
        """Higher is better."""
        return (
            self.row.get("status") == "completed",
            self.has_npz,
            -self.source_index,
            self.row.get("completed_at") or "",
        )


@dataclass
class MergeReport:
    per_source: dict = field(default_factory=dict)
    collisions: int = 0
    superseded: int = 0
    kept: int = 0
    kept_missing_npz: list = field(default_factory=list)
    copied_artefacts: int = 0

    def to_json(self) -> str:
        return json.dumps(self.__dict__, indent=2, default=str)


def merge(
    sources: list[tuple[Path, Path]],
    dest_db: Path,
    data_root: Path,
    copy_artefacts: bool = True,
) -> MergeReport:
    """Merge registries. ``sources`` = [(db_path, artefact_root)], priority order."""
    report = MergeReport()
    by_key: dict[tuple, list[_Candidate]] = {}
    all_uuids: dict[str, str] = {}
    cols: list[str] = []

    for i, (db_path, artefact_root) in enumerate(sources):
        rows, cols = _read_rows(db_path)
        report.per_source[str(db_path)] = {"rows": len(rows)}
        for row in rows:
            u = str(row[UUID_COL])
            if u in all_uuids:
                raise ValueError(f"uuid collision across sources: {u} in {all_uuids[u]} and {db_path}")
            all_uuids[u] = str(db_path)
            cand = _Candidate(row, i, str(db_path), artefact_root)
            cand.has_npz = _npz_exists(row, artefact_root)
            by_key.setdefault(_identifying_key(row), []).append(cand)

    winners: list[_Candidate] = []
    losers: list[_Candidate] = []
    for key, cands in by_key.items():
        cands.sort(key=_Candidate.rank, reverse=True)
        best, rest = cands[0], cands[1:]
        if best.row.get("status") == "completed":
            winners.append(best)
            losers.extend(rest)
        else:  # no completed run for this tuple: nothing canonical to keep
            losers.extend(cands)
        if rest:
            report.collisions += 1

    if dest_db.exists():
        raise FileExistsError(dest_db)
    _create_store_db(dest_db)
    con = sqlite3.connect(dest_db)
    try:
        with con:
            win_rows = []
            for w in winners:
                _canonical_paths(w.row, data_root)
                win_rows.append(w.row)
            _insert(con, "runs", win_rows, cols)

            sup_cols = cols + ["superseded_by_uuid", "source_db"]
            col_defs = ", ".join(f'"{c}"' for c in sup_cols)
            con.execute(f"CREATE TABLE runs_superseded ({col_defs})")
            sup_rows = []
            winner_by_key = {_identifying_key(w.row): str(w.row[UUID_COL]) for w in winners}
            for l in losers:
                r = dict(l.row)
                r["superseded_by_uuid"] = winner_by_key.get(_identifying_key(l.row))
                r["source_db"] = l.source_db
                sup_rows.append(r)
            _insert(con, "runs_superseded", sup_rows, sup_cols)
    finally:
        con.close()

    report.kept = len(winners)
    report.superseded = len(losers)

    for w in winners:
        src_dir = w.artefact_root / str(w.row["experiment_type"]) / str(w.row[UUID_COL])
        dst_dir = data_root / str(w.row["experiment_type"]) / str(w.row[UUID_COL])
        if copy_artefacts and w.has_npz and src_dir != dst_dir and not (dst_dir / "trace.npz").is_file():
            shutil.copytree(src_dir, dst_dir, dirs_exist_ok=True)
            report.copied_artefacts += 1
        if not (dst_dir / "trace.npz").is_file():
            report.kept_missing_npz.append(str(w.row[UUID_COL]))

    for i, (db_path, _) in enumerate(sources):
        report.per_source[str(db_path)]["kept"] = sum(1 for w in winners if w.source_index == i)
        report.per_source[str(db_path)]["superseded"] = sum(1 for l in losers if l.source_index == i)
    return report


# ---------------------------------------------------------------------------
# CLI


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("convert", help="copy one legacy registry into a current-wallow store")
    c.add_argument("--source", required=True)
    c.add_argument("--out", required=True)
    c.add_argument("--data-root", required=True, help="canonical artefact root for path rewriting")

    m = sub.add_parser("merge", help="merge N registries (priority = order) into one store")
    m.add_argument("--source", action="append", required=True, metavar="DB:ARTEFACT_ROOT")
    m.add_argument("--out", required=True)
    m.add_argument("--data-root", required=True)
    m.add_argument("--no-copy-artefacts", action="store_true")
    m.add_argument("--report", help="write the merge report JSON here")

    args = ap.parse_args(argv)
    if args.cmd == "convert":
        n = convert(Path(args.source), Path(args.out), Path(args.data_root))
        print(f"converted {n} rows -> {args.out}")
    else:
        sources = []
        for spec in args.source:
            db, root = spec.rsplit(":", 1)
            sources.append((Path(db), Path(root)))
        report = merge(
            sources, Path(args.out), Path(args.data_root),
            copy_artefacts=not args.no_copy_artefacts,
        )
        print(report.to_json())
        if args.report:
            Path(args.report).write_text(report.to_json())


if __name__ == "__main__":
    main()
