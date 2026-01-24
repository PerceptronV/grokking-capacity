# Competing Speeds of Memorisation and Generalisation Predict Grokking

Deep networks trained with gradient descent have been observed to exhibit 'grokking', or delayed generalisation, on small algorithmic datasets. One class of explanations frames grokking as the result of different pattern learning speeds, where gradient descent first learns fast patterns that may overfit, and only later learns slower patterns that generalise better. However, this has not led to testable frameworks for predicting when grokking should arise. In this work, we present a quantitative theory of grokking in modular arithmetic, demonstrating that the onset of grokking correlates strongly with the intersection of memorisation and generalisation speeds, where the speed by which a model memorises random data equals that by which it discovers an algorithmic solution. We make our statements rigorous by building on recent work in language model memorisation to establish information-theoretic estimates of model capacity and dataset complexity. Surprisingly, we find that smaller models do not grok even if they have enough capacity to memorise the training set. We argue this is because smaller models have slower memorisation speeds, biasing gradient descent towards first discovering the faster, generalising solution. We also demonstrate empirically that model capacity to dataset complexity ratio accurately predicts memorisation speed through an exponential relationship, offering a principled explanation for the previously observed phenomenon that larger models memorise faster. Our results support a three-stage picture of grokking: as model capacity increases above zero, we move from underfitting, to immediate generalisation, and finally to grokking. We provide evidence that memorisation and learning speeds are sufficient to quantitatively predict these regimes for grokking, and may be useful for understanding the generalisation behaviour of larger models on natural tasks.

## Reproducibility

To reproduce the data, run the following command:

```bash
python main.py
```

## License and Attribution

Unless otherwise stated, the files and code in this repository are licensed under the GNU GENERAL PUBLIC LICENSE (Version 3), Copyright (C) 2025 Yiding Song and Hanming Ye.

**Note:** the files [`data.py`](data.py) and [`models.py`](models.py) are adapted from the code by Amund Tveit (available at [adveit/torch_grokking](https://github.com/atveit/torch_grokking) under the MIT License), which itself is a PyTorch port of the original MLX code by Jason Stock (available at [stockeh/mlx-grokking](https://github.com/stockeh/mlx-grokking)). I have modified `data.py` to add different split types and random data generation, and left `models.py` untouched. The trainers used to finetune the neural network also takes inspiration from the code of Tveit and Stock.
