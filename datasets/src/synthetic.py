import numpy as np
from numpy.random import exponential, normal, poisson, uniform

DSIZE = (100, 1)


def generate_data(dsize=DSIZE):
    return {  # TODO: size change / mean change for same ds?
        'normal': [
            normal(loc=0, scale=1, size=dsize),
            normal(loc=0, scale=5, size=dsize),
            # normal(loc=5, scale=1, size=dsize),
        ],
        'uniform': [
            uniform(low=-1, high=1, size=dsize),
            uniform(low=-5, high=5, size=dsize),
        ],
        'poisson': [
            poisson(lam=1, size=dsize),
            poisson(lam=5, size=dsize),
        ],
        'exp': [
            exponential(scale=5, size=dsize),
            exponential(scale=10, size=dsize),
        ],
        # 'cauchy': [
        #     standard_cauchy(size=dsize),
        #     standard_cauchy(size=dsize) * 5,
        # ],
    }


def make_datasets(data, large=True, half=False):
    for i, fst in enumerate(data.keys()):
        for j, snd in enumerate(data.keys()):
            if half and j < i:
                continue
            intervals = [
                data[fst][0],
                data[fst][1] if fst == snd else data[snd][0],
                data[fst][0],
            ] + ([data[fst][1] if fst == snd else data[snd][0], data[fst][0]] if large else [])
            yield fst, snd, np.concatenate(intervals)
