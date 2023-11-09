import logging
from abc import ABC, abstractmethod

import numpy as np

logger = logging.getLogger(__name__)


# TODO: proper typing
class CPD_Algorithm(ABC):
    def __init__(self, mu: float):
        self.mu: float = mu

        self.change_points: list[int] = []
        self.t = 0
        self._step = False

        self.Y = self.Y_ref = self.Y_test = self.n_ref = self.n_test = self.k = self.s_history = self.t_list = None

    @abstractmethod
    def update(self, y, update_alpha=True):
        pass

    @abstractmethod
    def fit(self):
        pass

    @abstractmethod
    def calculate_S(self) -> float:
        pass

    def step(self, y):
        if not self._step:
            self.update(y)
            S = self.calculate_S()

            if np.abs(S) > self.mu:
                self._step = True
                self.s_history.append(np.nan)
                logger.info(f'change at {self.t}')
                self.change_points.append(self.t)
                return True

            else:
                self.s_history.append(S)
                return False

        else:
            self._step = False
            self.s_history.append(np.nan)
            self.update(y, update_alpha=False)
        return False

    def run(self, Y, n_ref, n_test, k):
        self.Y = Y
        self.n_ref = n_ref
        self.n_test = n_test
        self.k = k
        size = n_test + n_ref + k
        self.s_history = [np.nan] * (size)
        self.t_list = np.arange(0, size, 1)

        def take():
            if Y.shape[0] >= self.t + size:
                slice_ = Y[self.t : self.t + size]
                self.t += size
                # TODO: np.lib.stride_tricks.sliding_window_view ?
                M = np.array([slice_[i : i + k, :] for i in range(self.n_ref + self.n_test)])
                return M[: self.n_ref], M[self.n_ref : self.n_ref + self.n_test]
            else:
                self.t += size
                return None

        def train():
            if (slice_ := take()) is not None:
                self.Y_ref, self.Y_test = slice_[0], slice_[1]
                self.fit()
                return True
            return False

        def loop():
            train()
            while self.t < Y.shape[0]:
                y = Y[self.t]
                self.t += 1
                self.t_list = np.append(self.t_list, self.t)
                if self.step(y) and not train():
                    break

        loop()
        return self
