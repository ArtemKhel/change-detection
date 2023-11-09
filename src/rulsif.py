import logging

import numpy as np


logger = logging.getLogger('asdf')


# noinspection PyPep8Naming,PyAttributeOutsideInit
class RULSIF:
    def __init__(self, mu, lambda_, alpha=0.0, sigma=10):
        self.sigma = sigma  # TODO:
        self.rulsif_alpha = alpha
        self.mu = mu
        self.lambda_ = lambda_

        self.alpha = np.nan

        self.change_points = []
        self.t = 0
        self.first_run = True
        self.step_ = False

    def fit(self, Y_ref, Y_test):
        n_test = Y_test.shape[0]

        # TODO: check indices
        H = np.array(
            [
                [
                    (self.rulsif_alpha / n_test)
                    * sum(
                        self.calculate_K_sigma(self.Y_ref[j], self.Y_ref[row])
                        * self.calculate_K_sigma(self.Y_ref[j], self.Y_ref[col])
                        for j in range(n_test)
                    )
                    + ((1 - self.rulsif_alpha) / n_test)
                    * sum(
                        self.calculate_K_sigma(self.Y_test[j], self.Y_ref[row])
                        * self.calculate_K_sigma(self.Y_test[j], self.Y_ref[col])
                        for j in range(n_test)
                    )
                    for row in range(n_test)
                ]
                for col in range(n_test)
            ]
        )
        h = np.array(
            [
                [(1 / n_test) * sum(self.calculate_K_sigma(self.Y_ref[i], self.Y_ref[col]) for i in range(n_test))]
                for col in range(n_test)
            ]
        )

        try:
            self.alpha = np.linalg.inv(H + self.lambda_ * np.identity(H.shape[0])) @ h
        except np.linalg.LinAlgError as e:
            logger.error(e)
        self.w_hat = lambda Y: sum(self.alpha[i] * self.calculate_K_sigma(Y, self.Y_ref[i]) for i in range(n_test))

    # def loocv(self, Y_ref, Y_test, sigmas, lambdas):  # TODO: loocv
    #     pass

    def calculate_K_sigma(self, y, y_):
        return np.exp(-(np.linalg.norm(y - y_) ** 2) / (2 * self.sigma**2))

    def update(self, y, update_alpha=True):
        test_first = self.Y_test[np.newaxis, 0]
        self.Y_test = np.vstack([self.Y_test[1:], np.append(self.Y_test[-1][np.newaxis, 1:], [[y]], axis=1)])
        # if update_alpha:  # TODO: <--
        #     self.update_alpha()
        self.Y_ref = np.vstack((self.Y_ref[1:], test_first))

    # def update_alpha(self): # TODO: <--
    #     pass

    def calculate_S(self):
        return (
            (-self.rulsif_alpha / (2 * self.n_ref)) * sum((self.w_hat(self.Y_ref[i]) ** 2) for i in range(self.n_ref))
            - ((1 - self.rulsif_alpha) / (2 * self.n_test))
            * sum(self.w_hat(self.Y_test[i]) ** 2 for i in range(self.n_ref))
            + (1 / self.n_test) * sum(self.w_hat(self.Y_ref[i]) for i in range(self.n_test))
            - 0.5
        )

    def step(self, y):
        if not self.step_:
            self.update(y)
            S = self.calculate_S()[0]

            if np.abs(S) > self.mu:
                self.step_ = True
                self.s_history.append(np.nan)
                logger.info(f'change at {self.t}')
                self.change_points.append(self.t)
                return True

            else:
                self.s_history.append(S)
                return False

        else:
            self.step_ = False
            self.s_history.append(np.nan)
            self.update(y, update_alpha=False)
        return False

    def run(self, Y, n_ref, n_test, k):
        self.Y = Y
        self.n_ref = n_ref
        self.n_test = n_test
        self.k = k
        self.t = 0
        size = n_test + n_ref + k
        self.s_history = [np.nan] * (size)
        self.t_list = np.arange(0, size, 1)

        def take():
            if Y.shape[0] >= self.t + size:
                slice_ = Y[self.t : self.t + size]
                self.t += size
                M = np.array([slice_[i : i + k, :] for i in range(self.n_ref + self.n_test)])
                return M[: self.n_ref], M[self.n_ref : self.n_ref + self.n_test]
            else:
                self.t += size
                return None

        def train():
            if (slice_ := take()) is not None:
                self.Y_ref, self.Y_test = slice_[0], slice_[1]
                self.fit(self.Y_ref, self.Y_test)
                return True
            return False

        def loop():
            train()
            while self.t < Y.shape[0]:
                y = Y[self.t]
                self.t += 1
                self.t_list = np.append(self.t_list, self.t)
                if self.step(y):
                    if not train():
                        break

        loop()
        return self
