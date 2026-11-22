# How Large Must a Model Be to Grok?

This repository contains code for reproducing the experimens and data associated with the paper ['How Large Must a Model Be to Grok? A competition between memorisation and generalisation timescales'](https://arxiv.org/abs/2605.09724), by Yiding Song and Hanming Ye. It is published at the 40th Conference on Neural Information Processing Systems (NeurIPS 2026), Sydney, Australia.

**Abstract**

> Despite a known dependence between grokking and model parameter count, how model capacity shapes grokking remains an open question. Focusing on small Transformers and modular arithmetic, we give a quantitative account of the relationship between model capacity and grokking as a competition between two timescales: generalisation ($T_{\text{gen}}$) and memorisation ($T_{\text{mem}}$).
Building on prior formalisms for the empirical information capacity of Transformers, we define and measure $T_{\text{gen}}$ on modular arithmetic datasets and $T_{\text{mem}}$ on random-label datasets of equivalent complexity, and show that their intersection correlates with model sizes where consistent grokking begins.
We also observe empirical laws for $T_{\text{gen}}$ and $T_{\text{mem}}$ as models and datasets scale, motivating their forms theoretically and finding that our model of $T_{\text{mem}}$ is consistent with prior observations that larger models tend to memorise faster.
Moreover, extrapolating our empirical timescale laws predicted the existence of datasets for which no model of any size would consistently grok, a pattern supported on the modular arithmetic datasets of larger primes, demonstrating that our framework can make qualitative predictions out-of-distribution.

## Installation

In your micromamba environment of choice, run the following command to install the package in editable mode:

```bash
pip install -e .
```

This exposes the experiment entry points (`gc-capacity`, `gc-speed`, `gc-groks`), the suite dispatcher (`gc-dispatch`), the figure renderer (`gc-figures`), and the grouped workflow command (`gc-workflow`). The experiment runs database lives at `runs.db` (override with `GC_WALLOW_DB`); the schema is declared in `wallow.toml` and auto-creates on first use.

## Reproducibility

### Repository layout

- `configs/`: dispatcher suites; development recipes are grouped in subdirectories.
- `src/grokking_capacity/`: training, models, registry access, and reusable analysis.
- `scripts/`: two suite launchers and their command guide; see the
  [command guide](scripts/README.md).
- `tests/`: unit and integration checks using synthetic data and temporary stores.
- `docs/`: separate manuscript checkout, described above.

### Experiments and Data

## License and Attribution

Use the following BibTeX entry to cite this work:

```bibtex
@inproceedings{
  song2026grokking,
  title={How Large Must a Model Be to Grok? A competition between memorisation and generalisation timescales},
  author={Yiding Song and Hanming Ye},
  booktitle={The Fortieth Annual Conference on Neural Information Processing Systems},
  year={2026},
  url={https://openreview.net/forum?id=2kLkhhcpNe}
}
```

Unless otherwise stated, the files and code in this repository are licensed under the GNU GENERAL PUBLIC LICENSE (Version 3), Copyright (C) 2026 Yiding Song and Hanming Ye.

**Note:** the files [`modular.py`](src/grokking_capacity/data/modular.py) and [`transformer.py`](src/grokking_capacity/models/transformer.py) are adapted from the code by Amund Tveit (available at [adveit/torch_grokking](https://github.com/atveit/torch_grokking) under the MIT License), which itself is a PyTorch port of the original MLX code by Jason Stock (available at [stockeh/mlx-grokking](https://github.com/stockeh/mlx-grokking)). We have modified `modular.py` to add different split types and random data generation, and mostly left `transformer.py` untouched. The trainers used to finetune the neural network also takes inspiration from the code of Tveit and Stock.
