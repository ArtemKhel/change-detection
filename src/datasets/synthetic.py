import numpy as np

DSIZE = (100, 1)


def generate_data(dsize=DSIZE):
    return {  # TODO: size change / mean change for same ds?
        'normal': [
            np.random.normal(loc=0, scale=1, size=dsize),
            np.random.normal(loc=0, scale=5, size=dsize),
            # normal(loc=5, scale=1, size=dsize),
        ],
        'uniform': [
            np.random.uniform(low=-1, high=1, size=dsize),
            np.random.uniform(low=-5, high=5, size=dsize),
        ],
        'poisson': [
            np.random.poisson(lam=1, size=dsize),
            np.random.poisson(lam=5, size=dsize),
        ],
        'exp': [
            np.random.exponential(scale=5, size=dsize),
            np.random.exponential(scale=10, size=dsize),
        ],
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
