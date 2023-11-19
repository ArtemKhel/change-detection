from http import HTTPStatus

import numpy as np
import requests

__all__ = ['ZBOX_FILES', 'download_from_zboxfs']

URL = 'https://raw.githubusercontent.com/kik0908/zboxfs_speedtest/master/input/windows/'

ZBOX_FILES = [
    'finish_times_append_0',
    'finish_times_append_1',
    'finish_times_append_2',
    'finish_times_append_3',
    'finish_times_append_4',
    'finish_times_append_5',
    'finish_times_write_0',
    'finish_times_write_1',
    'finish_times_write_2',
    'finish_times_write_3',
    'finish_times_write_4',
    'finish_times_write_5',
    'finish_times_write_X',
    'finish_times_write_Y',
    'open_times_append_0',
    'open_times_append_1',
    'open_times_append_2',
    'open_times_append_3',
    'open_times_append_4',
    'open_times_append_5',
    'open_times_write_0',
    'open_times_write_1',
    'open_times_write_2',
    'open_times_write_3',
    'open_times_write_4',
    'open_times_write_5',
    'open_times_write_X',
    'open_times_write_Y',
    'write_times_append_0',
    'write_times_append_1',
    'write_times_append_2',
    'write_times_append_3',
    'write_times_append_4',
    'write_times_append_5',
    'write_times_write_0',
    'write_times_write_1',
    'write_times_write_2',
    'write_times_write_3',
    'write_times_write_4',
    'write_times_write_5',
    'write_times_write_X',
    'write_times_write_Y',
]


def download_from_zboxfs(name: str) -> np.array:
    with requests.get(url=URL + f'{name}.txt', stream=True) as response:
        if response.status_code == HTTPStatus.OK:
            return read_zboxfs(response.text)
        else:
            raise requests.HTTPError(f'Site returned {response.status_code}: {response.text}')


def read_zboxfs(text: str) -> np.array:
    return np.array(text.strip().split(), dtype=int)[:, np.newaxis]
