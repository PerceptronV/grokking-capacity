"""Historical command for the reusable dataset-comparison renderer."""
import sys
from grokking_capacity.experiments.registered import ROOT
from grokking_capacity.analysis.dataset_comparison_plot import main as render_main

if __name__ == '__main__':
    render_main(['--report', str(ROOT/'reports/p97_dataset_scaling'), *sys.argv[1:]])
