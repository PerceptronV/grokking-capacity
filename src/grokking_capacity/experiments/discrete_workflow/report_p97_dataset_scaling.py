"""Historical p97 command; reusable analysis lives in analysis.dataset_comparison."""
import sys
from grokking_capacity.experiments.registered import ROOT
from grokking_capacity.analysis.paired import summarize, crossing_brackets, analyze as analyze_pairs
from grokking_capacity.analysis.dataset_comparison import main as report_main


def analyze(rows, sizes, seeds, limit):
    return analyze_pairs(rows, sizes, seeds, limit, label_count=99)


def main():
    return report_main(['--run', str(ROOT/'results/discrete/p97-dataset-scaling-v1'),
                       '--out', str(ROOT/'reports/p97_dataset_scaling'), *sys.argv[1:]])


if __name__ == '__main__':
    main()
