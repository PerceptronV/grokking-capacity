"""Analysis commands share suite discovery without ingesting discrete recipes."""
import importlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import yaml

from grokking_capacity.analysis._cli import suite_configs


COMMANDS = ["cli", "bridge", "contraction", "error_decay", "kappa_stability",
            "rate_vs_capacity", "threshold_invariance", "mem_law"]


@pytest.fixture
def configs(tmp_path):
    directory = tmp_path / "configs"
    directory.mkdir()
    for name, spec in {
        "central": {"experiments": {"groks": {"dims": [32]}}},
        "alpha": {"experiments": {"speed": {"dims": [32]}}},
        "discrete_learning": {"defaults": {}, "runs": [{"model": "quantized"}]},
        "discrete_campaign": {"dataset": {}, "confirmation": {}},
    }.items():
        (directory / f"{name}.yaml").write_text(yaml.safe_dump(spec))
    return directory


@pytest.mark.parametrize("name", COMMANDS)
def test_all_selects_only_suites_and_preserves_output_layout(name, configs, tmp_path,
                                                          monkeypatch, capsys):
    module = importlib.import_module(f"grokking_capacity.analysis.{name}")
    output = tmp_path / "outputs"
    render = Mock()
    monkeypatch.setattr(module, "CONFIGS_DIR", configs)
    monkeypatch.setattr(module, "DEFAULT_OUT_ROOT", output)
    monkeypatch.setattr(module, "_render_one", render)
    assert module.main(["--all", "--out", str(tmp_path / "ignored"),
                        "--db", "local.db"]) == 0
    suffix = "" if name == "cli" else name
    assert [call.args[:2] for call in render.call_args_list] == [
        (configs / f"{suite}.yaml", output / suite / suffix)
        for suite in ["alpha", "central"]
    ]
    db_index = 3 if name == "cli" else 2
    assert all(call.args[db_index] == "local.db" for call in render.call_args_list)
    assert "--out is ignored with --all" in capsys.readouterr().err


@pytest.mark.parametrize("name", COMMANDS)
@pytest.mark.parametrize("explicit_out", [False, True])
def test_single_config_preserves_output_override(name, explicit_out, configs,
                                                tmp_path, monkeypatch):
    module = importlib.import_module(f"grokking_capacity.analysis.{name}")
    render = Mock()
    monkeypatch.setattr(module, "DEFAULT_OUT_ROOT", tmp_path / "outputs")
    monkeypatch.setattr(module, "_render_one", render)
    args = ["--config", str(configs / "central.yaml")]
    if explicit_out:
        args += ["--out", str(tmp_path / "custom")]
    assert module.main(args) == 0
    suffix = "" if name == "cli" else name
    expected = tmp_path / "custom" if explicit_out else tmp_path / "outputs/central" / suffix
    assert render.call_args.args[:2] == (configs / "central.yaml", expected)


def test_invalid_yaml_is_reported(configs):
    (configs / "broken.yaml").write_text("experiments: [unterminated")
    with pytest.raises(yaml.YAMLError):
        list(suite_configs(configs))


def test_stats_only_does_not_render_other_figures(tmp_path, monkeypatch):
    from grokking_capacity.analysis import cli
    view = SimpleNamespace(groups=[object()], swept_axes=[])
    monkeypatch.setattr(cli.ConfigView, "from_yaml", Mock(return_value=view))
    plots = Mock()
    stats = Mock(return_value=[])
    meta = Mock()
    monkeypatch.setattr(cli, "render_all", plots)
    monkeypatch.setattr(cli, "render_stats", stats)
    monkeypatch.setattr(cli, "write_meta", meta)
    output = tmp_path / "figures"
    cli._render_one(Path("central.yaml"), output, {"stats"}, None, False)
    plots.assert_not_called()
    stats.assert_called_once_with(view, output)
    meta.assert_called_once_with(view, output)
