"""Multiplicative-character spectra and common trace measurements."""
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import argparse
import numpy as np
import yaml


@lru_cache(maxsize=None)
def primitive_root(p):
    if p < 2 or any(p % d == 0 for d in range(2, int(p**.5)+1)):
        raise ValueError('Multiplicative-character analysis requires a prime')
    if p == 2:
        return 1
    factors, n, d = [], p-1, 2
    while d*d <= n:
        if n % d == 0:
            factors.append(d)
            while n % d == 0:
                n //= d
        d += 1
    if n > 1:
        factors.append(n)
    return next(g for g in range(2,p) if all(pow(g,(p-1)//q,p) != 1 for q in factors))


@lru_cache(maxsize=None)
def character_order(p):
    g = primitive_root(p)
    return tuple(pow(g,m,p) for m in range(p-1))


def spectral_power(weights, p):
    """Absolute power of nonconstant characters; preserve the original FFT order."""
    values = weights[list(character_order(p)), :].astype(np.float64)
    values -= values.mean(0, keepdims=True)
    coefficients = np.fft.rfft(values, axis=0)
    return (np.abs(coefficients[1:(p-1)//2+1, :])**2).sum(1)


@dataclass(frozen=True)
class CharacterTrace:
    epochs: np.ndarray
    power: np.ndarray
    train: np.ndarray
    validation: np.ndarray

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as data:
            trace = cls(data['epochs'],data['out_pow'],data['train_acc'],data['val_acc'])
        n = len(trace.epochs)
        if (not n or trace.power.ndim != 2 or trace.power.shape[0] != n
                or len(trace.train) != n or len(trace.validation) != n
                or np.any(np.diff(trace.epochs) <= 0)):
            raise ValueError('Character trace arrays must align on increasing epochs')
        if (not np.isfinite(trace.power).all() or np.any(trace.power < 0)
                or np.any(trace.power.sum(1) <= 0)):
            raise ValueError('Character power must be finite, nonnegative, and nonzero per epoch')
        return trace

    @property
    def share(self):
        return self.power/self.power.sum(1, keepdims=True)

    def cohorts(self, multiple=2):
        selected = self.share[-1] > multiple/self.power.shape[1]
        return np.flatnonzero(selected), np.flatnonzero(~selected)

    def first_epoch(self, values, threshold, *, strict=False):
        hits = np.flatnonzero(values > threshold if strict else values >= threshold)
        return int(self.epochs[hits[0]]) if len(hits) else None

    def takeoff(self, winners, multiple):
        if not len(winners):
            return None
        return self.first_epoch(self.share[:,winners].mean(1)*self.power.shape[1], multiple, strict=True)

    def window(self, start, end):
        if start is None or end is None:
            return np.zeros(len(self.epochs), dtype=bool)
        return (self.epochs >= start) & (self.epochs <= end)


def log_slopes(epochs, matrix, indices, mask, *, min_points):
    if mask.sum() < min_points or not len(indices):
        return None
    return float(np.mean([np.polyfit(epochs[mask],np.log(matrix[mask,i]+1e-300),1)[0]
                          for i in indices]))


def sweep_options(profile, argv=None):
    root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=f'Analyze the {profile} character sweep')
    parser.add_argument('--config',type=Path,default=root/'configs/analysis/character_sweeps.yaml')
    parser.add_argument('--input-dir',type=Path)
    parser.add_argument('--out',type=Path)
    args = parser.parse_args(argv)
    config = yaml.safe_load(args.config.read_text())
    inputs = args.input_dir or (args.config.parent/config['input_dir']).resolve()
    selected = config['profiles'][profile]
    output = args.out or inputs/selected['output']
    output.parent.mkdir(parents=True,exist_ok=True)
    return inputs,output,selected
