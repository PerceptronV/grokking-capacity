"""Compatibility command for the configured dropout_followup grid."""
from ..campaigns.cli_grid import main
from ..registered import ROOT

if __name__ == '__main__':
    main(default_config=ROOT/'configs/campaigns/dropout_followup.yaml',
         default_root=ROOT/'results/discrete/p97-dropout-followup-v2')
