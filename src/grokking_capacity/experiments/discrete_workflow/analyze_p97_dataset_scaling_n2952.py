"""Historical N=2952 command for the configurable dataset-slice comparison."""
import sys
from grokking_capacity.experiments.registered import ROOT
from grokking_capacity.analysis.dataset_slice import main

if __name__ == '__main__':
    main(['--report',str(ROOT/'reports/p97_dataset_scaling'),
          '--run',str(ROOT/'results/discrete/p97-dataset-scaling-v1'),
          '--size','2952',*sys.argv[1:]])
