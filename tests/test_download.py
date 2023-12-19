import numpy as np

from src.datasets.download import load_dataset


def test_load_dataset():
    data, comments = load_dataset('test')

    assert np.all(
        data
        == np.array(
            [
                [1, 2, 3],
                [4, 5, 6],
                [7, 8, 9],
            ]
        )
    )
    assert comments == {0: 'test', 1: 'comment'}
