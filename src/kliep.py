import logging
from src.cpd_algorithm import CPD_Algorithm

import numpy as np


logger = logging.getLogger('cpd')


# noinspection PyPep8Naming,PyAttributeOutsideInit
class KLIEP(CPD_Algorithm):
    # https://sci-hub.ru/10.1002/sam.10124
    # TODO: report changes later than expected, check update()?

    def __init__(self, mu, eta, lambda_, converge_iter=1000, epsilon=0.01, sigmas=None):
        super().__init__(mu)
        self.eta = eta
        self.lambda_ = lambda_
        self.converge_iter = converge_iter
        self.epsilon = epsilon
        self.sigmas = sigmas if sigmas is not None else [2**i for i in range(-3, 4)]

        self.alpha = np.nan

        self.first_run = True

    def _fit(self, Y_ref, Y_test, sigma=None):
        if sigma is None:
            sigma = self.sigma

        n_test = Y_test.shape[0]

        K = self.calculate_K(Y_test, sigma)
        b = self.calculate_b(Y_ref, Y_test, sigma)

        self.alpha = np.random.rand(n_test)
        for _ in range(self.converge_iter):
            alpha = self.alpha + self.epsilon * K @ (1 / (K @ self.alpha))
            alpha += (1 - b.T * alpha) @ b / (b.T @ b)
            alpha = np.maximum(0, alpha)
            alpha /= b.T @ alpha

            if np.linalg.norm(self.alpha - alpha) < self.epsilon:
                logger.info(f'converged in {_}')
                break

            self.alpha = alpha
        else:
            logger.warning('Alpha did not converge')

        w_hat = lambda Y: sum(self.alpha[i] * self.calculate_K_sigma(Y, Y_test[i], sigma) for i in range(n_test))
        return w_hat

    def _likelihood_cross_validation(self, Y_ref, Y_test):
        n_test = Y_test.shape[0]
        R = 4  # TODO: <--
        j_scores = dict()
        chunk_size = n_test // R
        chunks = [Y_test[chunk_size * i : chunk_size * (i + 1)] for i in range(R)]
        for sigma in self.sigmas:
            j_scores[sigma] = []
            for r, chunk in enumerate(chunks):
                other_chunks = np.vstack([c for i, c in enumerate(chunks) if i != r])
                w_hat = self._fit(Y_ref, other_chunks, sigma)
                score = sum(np.log(w_hat(y)) for y in chunk) / chunk_size
                j_scores[sigma].append(score)
            j_scores[sigma] = np.average(j_scores[sigma])
        sorted_scores = sorted([x for x in j_scores.items() if np.isfinite(x[1])], key=lambda x: x[1], reverse=True)
        self.sigma = sorted_scores[0][0]
        logger.info(f'lcv sigma: {self.sigma}')
        return self._fit(Y_ref, Y_test)

    def calculate_K_sigma(self, y, y_, sigma=None):
        if sigma is None:
            sigma = self.sigma
        return np.exp(-(np.linalg.norm(y - y_) ** 2) / (2 * sigma**2))

    def calculate_K(self, Y_test, sigma):
        n_test = Y_test.shape[0]
        return np.array(
            [[self.calculate_K_sigma(Y_test[i], Y_test[j], sigma) for i in range(n_test)] for j in range(n_test)]
        )

    def calculate_b(self, Y_ref, Y_test, sigma=None):
        if sigma is None:
            sigma = self.sigma
        n_ref = Y_ref.shape[0]
        n_test = Y_test.shape[0]
        return (
            np.array(
                [sum([self.calculate_K_sigma(Y_ref[i], Y_test[j], sigma) for i in range(n_ref)]) for j in range(n_test)]
            )
            / n_ref
        )

    def update(self, y, update_alpha=True):
        test_first = self.Y_test[np.newaxis, 0]
        self.Y_test = np.vstack([self.Y_test[1:], np.append(self.Y_test[-1][np.newaxis, 1:], [[y]], axis=1)])

        if update_alpha:
            self.update_alpha()

        self.Y_ref = np.vstack((self.Y_ref[1:], test_first))

    def update_alpha(self):
        c = sum([self.alpha[i] * self.calculate_K_sigma(self.Y_test[-1], self.Y_test[i]) for i in range(self.n_test)])

        self.alpha = np.append(self.alpha[1:] * (1 - self.eta * self.lambda_), self.eta / c)

        b = self.calculate_b(self.Y_ref, self.Y_test)
        self.alpha += (1 - b.T * self.alpha) @ b / (b.T @ b)
        self.alpha = np.maximum(0, self.alpha)
        self.alpha /= b.T @ self.alpha

    def calculate_S(self):
        return sum([np.log(self.w_hat(self.Y_test[i])) for i in range(self.n_test)]) / self.n_test

    def fit(self):
        if self.first_run:
            self.first_run = False
            self.w_hat = self._likelihood_cross_validation(self.Y_ref, self.Y_test)
        else:
            self.w_hat = self._fit(self.Y_ref, self.Y_test)


if __name__ == '__main__':
    k = KLIEP(0, 0, 0)
