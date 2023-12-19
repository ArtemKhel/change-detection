import shutil
import tempfile
from http import HTTPStatus
from pathlib import Path

import numpy as np
import requests

from src.datasets.config import DATA_DIR, DATASET_URL

__all__ = [
    'load_dataset',
]


def download(name: str) -> Path:
    with requests.get(url=f'{DATASET_URL}/{name}.tar.xz', stream=True) as response:
        if response.status_code == HTTPStatus.OK:
            unpack_path = DATA_DIR / name
            with tempfile.NamedTemporaryFile('wb') as archive:
                shutil.copyfileobj(response.raw, archive)
                shutil.unpack_archive(archive.name, unpack_path, format='xztar')

            return unpack_path / f'{name}.txt'
        else:
            raise requests.HTTPError(f'Site returned {response.status_code}: {response.text}')


def parse(path: Path) -> tuple[np.array, dict[int, str]]:
    with open(path) as file:
        data = []
        comments = {}
        for i, line in enumerate(file.readlines()):
            vals, *comment = line.split('|', maxsplit=1)
            data.append(list(map(float, vals.split())))
            if comment:
                comments[i] = comment[0].strip()

        return np.array(data, dtype=float), comments


def load_dataset(name: str) -> tuple[np.array, dict[int, str]]:
    """
    Download dataset
    :param name: name of the dataset
    :return:
    """
    return parse(download(name))
