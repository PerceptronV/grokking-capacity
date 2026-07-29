#!/usr/bin/env bash
# Fix CUDA on this node and relaunch the theory-plan sweeps.
#
# Diagnosis (2026-07-29): torch reports "Error 802: system not yet
# initialized". This node is a 2-GPU slice of an HGX H100 machine: the
# H100 SXM5 GPUs' fabric state is stuck "In Progress" because no NVSwitch
# is passed through, and nvidia-fabricmanager correctly refuses to start
# ("Nothing to do"). The documented fix for switchless slices is to
# reload the driver with NVLink disabled (our jobs are all single-GPU and
# never use NVLink).
#
# Run as:  sudo scripts/fix_gpu_and_relaunch.sh
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
REAL_USER="${SUDO_USER:-$USER}"
REAL_HOME="$(eval echo "~$REAL_USER")"

echo "== 1. Reload nvidia driver with NVLink disabled =="
# nvidia-persistenced holds /dev/nvidia*; nvidia_peermem holds the module.
systemctl stop nvidia-persistenced || true
for m in nvidia_peermem nvidia_uvm nvidia_drm nvidia_modeset; do
  lsmod | grep -q "^$m " && modprobe -r "$m"
done
modprobe -r nvidia || {
  echo "modprobe -r nvidia failed — check holders: fuser -v /dev/nvidia*; lsmod | grep nvidia"; exit 1; }
modprobe nvidia NVreg_NvLinkDisable=1
modprobe nvidia_uvm
systemctl start nvidia-persistenced || true
nvidia-smi --query-gpu=index,name --format=csv,noheader

echo "== 2. Verify CUDA from torch =="
sudo -u "$REAL_USER" python3 - <<'EOF'
import torch
torch.zeros(8).cuda()
print("CUDA OK, devices:", torch.cuda.device_count())
EOF

echo "== 3. Clean up the aborted CPU-run registries and their artefacts =="
sudo -u "$REAL_USER" python3 - <<EOF
import os, shutil, sqlite3
repo = "$REPO"
home = "$REAL_HOME"
for db in ("batch_sweep_runs.db", "sgd_control_runs.db", "m0_runs.db"):
    path = os.path.join(home, db)
    if not os.path.exists(path):
        continue
    con = sqlite3.connect(path)
    try:
        rows = con.execute("select uuid, experiment_type from runs").fetchall()
    except sqlite3.OperationalError:
        rows = []
    for uuid, etype in rows:
        d = os.path.join(repo, "data", etype, uuid)
        if os.path.isdir(d):
            shutil.rmtree(d)
    con.close()
    os.remove(path)
    print("removed", path, f"({len(rows)} aborted rows + artefact dirs)")
EOF

echo "== 4. Relaunch the three tracks (NOT the p=197 sweep — that stays"
echo "      gated on the OpenReview pre-registration comment being posted) =="
cd "$REPO"
sudo -u "$REAL_USER" bash -c "
  cd '$REPO'
  GC_WALLOW_DB='$REAL_HOME/batch_sweep_runs.db' nohup gc-dispatch \
      --config configs/batch_size_sweep.yaml > logs/batch_sweep_dispatch.log 2>&1 &
  echo \"batch sweep dispatch pid \$!\"
  CUDA_VISIBLE_DEVICES=0 nohup scripts/run_sgd_control.sh > logs/sgd_control.log 2>&1 &
  echo \"sgd control pid \$!\"
  CUDA_VISIBLE_DEVICES=1 nohup scripts/run_m0_selection_runs.sh > logs/m0_selection_runs.log 2>&1 &
  echo \"m0 selection pid \$!\"
"
echo "Done. Watch: tail -f logs/batch_sweep_dispatch.log"
echo "After posting the pre-registration comment, launch the p=197 sweep with:"
echo "  GC_WALLOW_DB=\$HOME/p197_runs.db nohup gc-dispatch --config configs/p197_forecast.yaml > logs/p197_dispatch.log 2>&1 &"
