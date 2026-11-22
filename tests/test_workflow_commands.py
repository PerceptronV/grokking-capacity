"""Exercise command routing without starting experiments."""
from pathlib import Path
import sys

import pytest

from grokking_capacity import workflows
from grokking_capacity.utils.workflow_paths import ROOT, SOURCES, frozen_source


def test_scripts_directory_stays_within_three_source_files():
    files = [p for p in (ROOT / "scripts").rglob("*")
             if p.is_file() and "__pycache__" not in p.parts]
    assert len(files) <= 3


def test_catalogue_paths_exist():
    for commands in workflows.catalog().values():
        for entry in commands.values():
            assert (ROOT / entry["path"]).is_file()


def test_listing_does_not_execute_workflows(monkeypatch, capsys):
    def unexpected(*args, **kwargs):
        pytest.fail("Listing executed a workflow")
    monkeypatch.setattr(workflows.runpy, "run_module", unexpected)
    monkeypatch.setattr(workflows.runpy, "run_path", unexpected)
    workflows.main(["--list", "capacity"])
    output = capsys.readouterr().out
    assert "analyse_matched_capacity" in output
    assert "discrete:" not in output


@pytest.mark.parametrize("group,command,runner", [
    ("capacity", "analyse_matched_capacity", "run_module"),
    ("discrete", "run_p97_capacity_scaling", "run_module"),
])
def test_routing_preserves_arguments_and_restores_process_state(
        group, command, runner, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    original_args = sys.argv
    forwarded = ["--output", "a path; $(literal)", "--flag"]
    def execute(target, *, run_name):
        assert run_name == "__main__"
        assert sys.argv[1:] == forwarded
        assert Path.cwd() == ROOT
        if runner == "run_module":
            assert target.startswith("grokking_capacity.")
        else:
            assert Path(target) == ROOT / "scripts/run_p97_capacity_scaling.py"
        raise RuntimeError("simulated failure")
    monkeypatch.setattr(workflows.runpy, runner, execute)
    with pytest.raises(RuntimeError, match="simulated failure"):
        workflows.main([group, command, "--", *forwarded])
    assert Path.cwd() == tmp_path
    assert sys.argv is original_args


def test_unknown_command_does_not_execute(monkeypatch):
    monkeypatch.setattr(workflows.runpy, "run_module", lambda *a, **k: pytest.fail("Executed"))
    with pytest.raises(SystemExit) as failure:
        workflows.main(["discrete", "missing_command"])
    assert failure.value.code == 2


def test_frozen_workers_resolve_both_layouts(tmp_path):
    name = "run_cone_ten_seeds.py"
    packaged = tmp_path / "source" / SOURCES[Path(name).stem]
    packaged.parent.mkdir(parents=True)
    packaged.write_text("new snapshot")
    assert frozen_source(tmp_path, name) == packaged
    historical = tmp_path / "source/scripts" / name
    historical.parent.mkdir(parents=True)
    historical.write_text("original snapshot")
    assert frozen_source(tmp_path, name) == historical


def test_missing_frozen_worker_never_falls_back_to_live_code(tmp_path):
    with pytest.raises(FileNotFoundError):
        frozen_source(tmp_path, "run_cone_ten_seeds.py")
