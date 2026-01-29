# Competing Speeds of Memorisation and Generalisation Predict Grokking

Deep networks trained with gradient descent have been observed to exhibit `grokking', or delayed generalisation, on small algorithmic datasets.
It has been conjectured that grokking is the result of different pattern learning speeds, where gradient descent first learns fast patterns that may overfit, and only later learns slower patterns that generalise better. 
In this work, we formalise this conjecture by establishing information-theoretic estimates of model capacity and dataset complexity. We demonstrate that the onset of grokking correlates strongly with the intersection of memorisation and generalisation speeds, where the time taken by a model to find an algorithmic solution equals that required to memorise a dataset of equivalent complexity.
Surprisingly, we find that smaller models do not grok even if they have enough capacity to memorise the training set. We argue this is because smaller models have slower memorisation speeds, biasing gradient descent towards first discovering the faster, generalising solution.
Our experiments provide evidence that memorisation and learning speeds are sufficient to quantitatively model grokking, and may be useful for understanding the generalisation behaviour of larger models on natural tasks.

## Reproducibility

To reproduce the data, run the following command:

```bash
python main.py
```

## License and Attribution

Unless otherwise stated, the files and code in this repository are licensed under the GNU GENERAL PUBLIC LICENSE (Version 3), Copyright (C) 2025 Yiding Song and Hanming Ye.

**Note:** the files [`data.py`](data.py) and [`models.py`](models.py) are adapted from the code by Amund Tveit (available at [adveit/torch_grokking](https://github.com/atveit/torch_grokking) under the MIT License), which itself is a PyTorch port of the original MLX code by Jason Stock (available at [stockeh/mlx-grokking](https://github.com/stockeh/mlx-grokking)). We have modified `data.py` to add different split types and random data generation, and left `models.py` untouched. The trainers used to finetune the neural network also takes inspiration from the code of Tveit and Stock.
