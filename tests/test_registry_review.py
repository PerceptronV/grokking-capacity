"""Registry consolidation preserves source evidence and uses one output transaction."""
from contextlib import closing
from pathlib import Path
import sqlite3
import subprocess

import pytest

from grokking_capacity.registry import merge, provenance
from grokking_capacity.registry.identifying import IDENTIFYING_FIELDS
from grokking_capacity.registry.store import get_schema


def row(uuid, *, dim=8, status='completed', **annotations):
    values = {name: get_schema().fields[name].default for name in IDENTIFYING_FIELDS}
    values.update(experiment_type='groks', p=13, n_samples=78, dim=dim,
                  uuid=uuid, status=status, created_at='2026-01-01 00:00:00',
                  updated_at='2026-01-01 00:00:00', **annotations)
    return values


def source(path, rows, *, omit=(), legacy=False):
    columns = list(dict.fromkeys(key for item in rows for key in item if key not in omit))
    stored = ['run_uuid' if legacy and key == 'uuid' else key for key in columns]
    with closing(sqlite3.connect(path)) as con, con:
        con.execute('CREATE TABLE runs (' + ', '.join(stored) + ')')
        con.executemany('INSERT INTO runs VALUES (' + ', '.join('?' for _ in stored) + ')',
                        [[item.get(key) for key in columns] for item in rows])
    return path


def records(path, table='runs'):
    with closing(sqlite3.connect(path)) as con:
        con.row_factory = sqlite3.Row
        return [dict(item) for item in con.execute('SELECT * FROM ' + table)]


def trace(root, uuid, content):
    path = root / 'groks' / uuid / 'trace.npz'
    path.parent.mkdir(parents=True)
    path.write_bytes(content)
    return path


def test_merge_preserves_annotation_union_precedence_and_sources(isolated_repo):
    root = isolated_repo
    first = source(root / 'first?#.db', [
        row('a1', dim=8, status='failed', host='source-a'),
        row('a2', dim=16, host='source-a'),
        row('a3', dim=32, host='source-a'),
        row('a4', dim=64, status='failed', host='source-a'),
    ], legacy=True)
    second = source(root / 'second.db', [
        row('b1', dim=8, final_loss=.1),
        row('b2', dim=16, final_loss=.2),
        row('b3', dim=32, final_loss=.3),
    ])
    a, b = root / 'artifacts-a', root / 'artifacts-b'
    trace(a, 'a1', b'failed checkpoint')
    trace(a, 'a3', b'priority checkpoint')
    trace(b, 'b2', b'completed checkpoint')
    trace(b, 'b3', b'second priority checkpoint')
    before = {path: path.read_bytes() for path in [first, second]}
    out, artifacts = root / 'merged.db', root / 'canonical'
    report = merge.merge([(first, a), (second, b)], out, artifacts)
    winners = {item['uuid']: item for item in records(out)}
    losers = {item['uuid']: item for item in records(out, 'runs_superseded')}
    assert set(winners) == {'b1', 'b2', 'a3'}
    assert set(losers) == {'a1', 'a2', 'a4', 'b3'}
    assert winners['a3']['host'] == 'source-a'
    assert winners['b2']['final_loss'] == .2
    assert winners['b2']['host'] is None
    assert losers['a1']['host'] == 'source-a'
    assert losers['b3']['final_loss'] == .3
    assert losers['b3']['superseded_by_uuid'] == 'a3'
    assert losers['a4']['superseded_by_uuid'] is None
    assert winners['a3']['npz_path'] == str(artifacts / 'groks/a3/trace.npz')
    assert (artifacts / 'groks/a3/trace.npz').read_bytes() == b'priority checkpoint'
    assert (artifacts / 'groks/b2/trace.npz').read_bytes() == b'completed checkpoint'
    assert report.kept == 3 and report.superseded == 4 and report.collisions == 3
    assert report.copied_artefacts == 2 and report.kept_missing_npz == ['b1']
    assert all(path.read_bytes() == original for path, original in before.items())


def test_convert_preserves_legacy_uuid_and_destination_defaults(isolated_repo, monkeypatch):
    root = isolated_repo
    before = source(root / 'old.db', [row('legacy42')], omit=('mup_base',), legacy=True)
    original = before.read_bytes()
    monkeypatch.setenv('WALLOW_JOURNAL_MODE', 'WAL')
    out = root / 'converted.db'
    assert merge.convert(before, out, root / 'canonical') == 1
    converted, = records(out)
    assert converted['uuid'] == 'legacy42' and converted['mup_base'] == 0
    assert converted['npz_path'] == str(root / 'canonical/groks/legacy42/trace.npz')
    assert before.read_bytes() == original
    import os
    assert os.environ['WALLOW_JOURNAL_MODE'] == 'WAL'
    with closing(sqlite3.connect(out)) as con:
        assert con.execute('PRAGMA journal_mode').fetchone()[0] == 'delete'


@pytest.mark.parametrize('omit,extra', [(('mup_base',), {}), ((), {'unsupported_result': 1})])
def test_merge_rejects_incompatible_sources_before_creating_output(isolated_repo, omit, extra):
    source_path = source(isolated_repo / 'source.db', [row('a', **extra)], omit=omit)
    output = isolated_repo / 'destination.db'
    with pytest.raises(ValueError, match='Incompatible registry columns'):
        merge.merge([(source_path, isolated_repo)], output, isolated_repo / 'canonical')
    assert not output.exists()


def test_duplicate_uuids_are_rejected_before_output_creation(isolated_repo):
    a = source(isolated_repo / 'a.db', [row('duplicate')])
    b = source(isolated_repo / 'b.db', [row('duplicate', dim=16)])
    output = isolated_repo / 'out.db'
    with pytest.raises(ValueError, match='uuid collision'):
        merge.merge([(a, isolated_repo), (b, isolated_repo)], output, isolated_repo)
    assert not output.exists()


def test_existing_destination_is_never_replaced(isolated_repo):
    src = source(isolated_repo / 'source.db', [row('a')])
    output = isolated_repo / 'existing.db'
    output.write_bytes(b'user-owned content')
    with pytest.raises(FileExistsError):
        merge.convert(src, output, isolated_repo)
    assert output.read_bytes() == b'user-owned content'


def test_failed_conversion_removes_only_its_new_output(isolated_repo, monkeypatch):
    src = source(isolated_repo / 'source.db', [row('a')])
    before = src.read_bytes()
    output = isolated_repo / 'out.db'
    original_insert = merge._insert
    def fail_after_insert(*args, **kwargs):
        original_insert(*args, **kwargs)
        raise RuntimeError('simulated insertion failure')
    monkeypatch.setattr(merge, '_insert', fail_after_insert)
    with pytest.raises(RuntimeError, match='simulated insertion failure'):
        merge.convert(src, output, isolated_repo)
    assert not output.exists() and src.read_bytes() == before
    assert not Path(str(output) + '-journal').exists()


def test_empty_sources_are_rejected_without_output(isolated_repo):
    output = isolated_repo / 'out.db'
    with pytest.raises(ValueError, match='At least one source'):
        merge.merge([], output, isolated_repo)
    assert not output.exists()


def test_provenance_is_anchored_to_code_repository(tmp_path, monkeypatch):
    code_root = tmp_path / 'code'
    other = tmp_path / 'different-cwd'
    other.mkdir()
    monkeypatch.chdir(other)
    monkeypatch.setattr(provenance, '_repo_root', lambda: code_root)
    responses = iter([b'abc123\n', b'', b' M source.py\n'])
    def git(command, *, cwd, stderr):
        assert command[0] == 'git' and cwd == code_root
        assert stderr == subprocess.DEVNULL
        return next(responses)
    monkeypatch.setattr(provenance.subprocess, 'check_output', git)
    assert provenance._git_short_hash() == 'abc123'
    assert provenance._git_dirty() is False
    assert provenance._git_dirty() is True
    def unavailable(*args, **kwargs):
        raise OSError('git unavailable')
    monkeypatch.setattr(provenance.subprocess, 'check_output', unavailable)
    assert provenance._git_short_hash() is None
    assert provenance._git_dirty() is None


def test_source_connections_enforce_read_only(isolated_repo):
    path = source(isolated_repo / 'source.db', [row('a')])
    with closing(merge._connect_ro(path)) as con:
        with pytest.raises(sqlite3.OperationalError, match='readonly'):
            con.execute("UPDATE runs SET status='failed'")
    assert records(path)[0]['status'] == 'completed'


def test_identifying_field_contract_matches_schema():
    assert set(IDENTIFYING_FIELDS) == get_schema().identifying
