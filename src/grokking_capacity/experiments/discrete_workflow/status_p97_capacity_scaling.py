"""Compatibility command for campaign status; accepts --root."""
from ..registered import ROOT
from ..campaigns.status import main

if __name__ == '__main__':
    main(default_root=ROOT/'results/discrete/p97-capacity-scaling-v1')
