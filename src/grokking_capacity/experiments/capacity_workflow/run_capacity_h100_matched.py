"""Compatibility command for the configured matched_capacity grid."""
from ..campaigns.cli_grid import main
from ..registered import ROOT

if __name__ == '__main__':
    main(default_config=ROOT/'configs/campaigns/matched_capacity.yaml',
         default_root=ROOT/'results/capacity_matched-v2')
