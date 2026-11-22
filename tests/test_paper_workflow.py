"""Check workflow dependencies and the boundary between analysis and publication."""
from unittest.mock import Mock

import pytest

from grokking_capacity.analysis.paper import __main__ as workflow
from grokking_capacity.analysis.paper import crossings, models, statistics, supplement


def test_stages_use_requested_output_and_dependency_order(tmp_path, monkeypatch):
    calls = []
    for module, name, stage in [(crossings, "run_analysis", "crossings"),
                                (statistics, "run", "statistics"),
                                (models, "run", "models"),
                                (supplement, "run", "supplement")]:
        monkeypatch.setattr(module, name, lambda out, _stage=stage, **kw:
                            calls.append((_stage, out, kw)))
    workflow.run(tmp_path, stages=("supplement", "models", "statistics", "crossings"),
                 bootstraps=17)
    assert [stage for stage, _, _ in calls] == list(workflow.STAGES)
    assert all(out == tmp_path / stage for stage, out, _ in calls)
    assert calls[0][2] == {"bootstraps": 17}
    assert calls[2][2] == calls[3][2] == {"statistics": tmp_path / "statistics"}


def test_selected_stage_does_not_run_others(tmp_path, monkeypatch):
    selected = Mock()
    monkeypatch.setattr(statistics, "run", selected)
    for module, name in [(crossings, "run_analysis"), (models, "run"), (supplement, "run")]:
        monkeypatch.setattr(module, name, lambda *a, **k: pytest.fail("Unexpected stage"))
    workflow.main(["--stage", "statistics", "--out", str(tmp_path)])
    selected.assert_called_once_with(tmp_path / "statistics")


@pytest.mark.parametrize("kwargs", [{"stages": ["unknown"]}, {"bootstraps": 0}])
def test_invalid_options_do_not_start_analysis(tmp_path, monkeypatch, kwargs):
    monkeypatch.setattr(crossings, "run_analysis", lambda *a, **k: pytest.fail("Started analysis"))
    with pytest.raises(ValueError):
        workflow.run(tmp_path, **kwargs)
