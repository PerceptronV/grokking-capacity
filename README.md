# Model Capacity Determines Grokking through Competing Memorisation and Generalisation Speeds

Existing accounts of grokking explain the phenomena in terms of mechanistic frameworks such as circuit efficiency or lazy-to-rich transitions. However, despite a known dependence between grokking and model size, how model capacity shapes grokking remains an open question. We give an information-theoretic account of this relationship on the task of modular arithmetic, showing that grokking does not immediately occur when a model becomes large enough to memorise the training set, but rather emerges as the outcome of a competition between two measurable timescales: a memorisation speed $T_{\text{mem}}(P)$ and a generalisation speed $T_{\text{gen}}(P)$, both of which are functions of model parameter count $P$. Adapting the information capacity framework of Morris et al. (2025), we estimate $T_{\text{mem}}(P)$ on random-label data of equivalent complexity and $T_{\text{gen}}(P)$ on the modular task itself, and show that grokking emerges close to the parameter scale where these timescales intersect. The framework also suggests an empirical model for predicting memorisation speed given model capacity and dataset complexity, recovering the previously reported empirical observation that larger models memorise faster. Overall, we motivate the formalisation of different learning timescales as important abstractions to study when explaining how model capacity shapes grokking on algorithmic tasks.

## Installation

In your micromamba environment of choice, run the following command to install the package in editable mode:

```bash
pip install -e .
```

This exposes the experiment entry points (`gc-capacity`, `gc-speed`, `gc-groks`), the suite dispatcher (`gc-dispatch`), and the figure renderer (`gc-figures`). The experiment runs database lives at `runs.db` (override with `GC_WALLOW_DB`); the schema is declared in `wallow.toml` and auto-creates on first use.

## Reproducibility

The paper's data is produced by a set of YAML-driven suites under `configs/`. Each suite is dispatched in parallel across the available GPUs by `gc-dispatch`.

To reproduce every suite end-to-end on a single multi-GPU node, run the wrapper script from the repo root with your micromamba environment name:

```bash
scripts/replicate.sh <micromamba-env>
```

This launches a detached `tmux` session that runs all ten suites (`central`, `weight_decay_sweep`, `alpha_sweep`, `dropout_sweep`, `lr_sweep`, `init_scale_sweep`, `task_add`, `depth_scaling`, `heads_sweep`, `task_mul`) sequentially.

Once the wallow database is populated, render the figures off it (one folder per config under `figures/`):

```bash
gc-figures --all
```

To render a single config or restrict to a figure family, see `gc-figures --help`.

## License and Attribution

Use the following BibTeX entry to cite this work:

```bibtex
@article{song2026grokking,
  title={Model Capacity Determines Grokking through Competing Memorisation and Generalisation Speeds},
  author={Song, Yiding and Ye, Hanming}
  year={2026}
}
```

Unless otherwise stated, the files and code in this repository are licensed under the GNU GENERAL PUBLIC LICENSE (Version 3), Copyright (C) 2026 Yiding Song and Hanming Ye.

**Note:** the files [`modular.py`](src/grokking_capacity/data/modular.py) and [`transformer.py`](src/grokking_capacity/models/transformer.py) are adapted from the code by Amund Tveit (available at [adveit/torch_grokking](https://github.com/atveit/torch_grokking) under the MIT License), which itself is a PyTorch port of the original MLX code by Jason Stock (available at [stockeh/mlx-grokking](https://github.com/stockeh/mlx-grokking)). We have modified `modular.py` to add different split types and random data generation, and mostly left `transformer.py` untouched. The trainers used to finetune the neural network also takes inspiration from the code of Tveit and Stock.
