"""Run the replication orchestration against completed synthetic stage outputs."""
import json
from types import SimpleNamespace

from grokking_capacity.experiments.discrete_workflow import run_discrete_replication as pipeline


def test_replication_runs_every_stage_through_importable_modules(tmp_path, monkeypatch):
    output = tmp_path / "replication"
    gate = tmp_path / "gate"
    gate.mkdir()
    evidence = {"path": str(gate), "summary_sha256": "test"}
    (gate / "audit.json").write_text(json.dumps({"arithmetic_gate_passed": True, "gate": evidence}))
    monkeypatch.setattr(pipeline, "check_gate", lambda path: evidence)
    monkeypatch.setattr(pipeline, "configured_paths", lambda config, root: [root / "s42"])
    for directory, name, value in [
        ("random-learning/s42", "summary.json", {"metrics": {"train_99": 100}, "final": {"train_acc": 1.}}),
        ("capacity-B/s42", "manifest.json", {"config": {"p": 97}}),
        ("confirmation", "selection.json", {"runs": [{"name": "A_s43"}]}),
    ]:
        path = output / directory
        path.mkdir(parents=True)
        (path / name).write_text(json.dumps(value))
    commands = []
    def launch(command, **kwargs):
        assert command[1:3] == ["-u", "-m"]
        commands.append(command[3])
        return SimpleNamespace(pid=123, poll=lambda: 0, returncode=0)
    monkeypatch.setattr(pipeline.subprocess, "Popen", launch)
    args = SimpleNamespace(output=output, gate=gate, resume=False, report=tmp_path / "report",
                           random_learning=tmp_path / "random.yaml",
                           a_capacity=tmp_path / "a.yaml", b_capacity=tmp_path / "b.yaml",
                           c_capacity=tmp_path / "c.yaml", a_device="cpu", b_device="cpu")
    pipeline.replicate(args)
    prefix = "grokking_capacity.experiments."
    assert commands == [prefix + name for name in [
        "discrete", "discrete_workflow.audit_discrete_capacity", "discrete", "discrete",
        "discrete_workflow.audit_discrete_capacity", "discrete_workflow.capacity",
        "discrete_workflow.confirmation", "discrete", "discrete_workflow.audit_discrete",
        "discrete_workflow.audit_discrete_capacity", "discrete_workflow.report",
    ]]
    assert json.loads((output / "pipeline.json").read_text())["phase"] == "complete"
