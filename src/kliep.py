import logging

import matplotlib.pyplot as plt
import numpy as np

# from src.data_generation import Square_Wave, Saw_Wave, Triangle_Wave, VaryingAmplitudeNoise

logger = logging.getLogger('asdf')


# noinspection PyPep8Naming,PyAttributeOutsideInit
class KLIEP:
    # https://sci-hub.ru/10.1002/sam.10124
    # TODO: report changes later than expected, check update()?

    def __init__(self, mu, eta, lambda_, converge_iter=1000, epsilon=0.01, sigmas=None):
        self.mu = mu
        self.eta = eta
        self.lambda_ = lambda_
        self.converge_iter = converge_iter
        self.epsilon = epsilon
        self.sigmas = sigmas if sigmas is not None else [2 ** i for i in range(-3, 4)]

        self.alpha = np.nan

        self.change_points = []
        self.t = 0
        self.first_run = True
        self.step_ = False

    def fit(self, Y_ref, Y_test, sigma=None):
        if sigma is None:
            sigma = self.sigma

        n_ref = Y_ref.shape[0]
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

    def likelihood_cross_validation(self, Y_ref, Y_test):
        # TODO: doesn't work?
        n_ref = Y_ref.shape[0]
        n_test = Y_test.shape[0]
        R = 4  # TODO: <--
        j_scores = dict()
        chunk_size = n_test // R
        chunks = [Y_test[chunk_size * i:chunk_size * (i + 1)] for i in range(R)]
        for sigma in self.sigmas:
            j_scores[sigma] = []
            for r, chunk in enumerate(chunks):
                other_chunks = np.vstack([c for i, c in enumerate(chunks) if i != r])
                w_hat = self.fit(Y_ref, other_chunks, sigma)
                score = sum(np.log(w_hat(y)) for y in chunk) / chunk_size
                j_scores[sigma].append(score)
            j_scores[sigma] = np.average(j_scores[sigma])
        sorted_scores = sorted([x for x in j_scores.items() if np.isfinite(x[1])], key=lambda x: x[1], reverse=True)
        self.sigma = sorted_scores[0][0]
        logger.info(f'lcv sigma: {self.sigma}')
        return self.fit(Y_ref, Y_test)

    def calculate_K_sigma(self, y, y_, sigma=None):
        if sigma is None:
            sigma = self.sigma
        return np.exp(-(np.linalg.norm(y - y_) ** 2) / (2 * sigma ** 2))

    def calculate_K(self, Y_test, sigma):
        n_test = Y_test.shape[0]
        return np.array(
            [
                [
                    self.calculate_K_sigma(Y_test[i], Y_test[j], sigma)
                    for i in range(n_test)
                ] for j in range(n_test)
            ]
        )

    def calculate_b(self, Y_ref, Y_test, sigma=None):
        if sigma is None:
            sigma = self.sigma
        n_ref = Y_ref.shape[0]
        n_test = Y_test.shape[0]
        return (
            np.array([
                sum(
                    [self.calculate_K_sigma(Y_ref[i], Y_test[j], sigma)
                     for i in range(n_ref)])
                for j in range(n_test)
            ]) / n_ref
        )

    def update(self, y, update_alpha=True):
        test_first = self.Y_test[np.newaxis, 0]
        self.Y_test = np.vstack([self.Y_test[1:], np.append(self.Y_test[-1][np.newaxis, 1:], [[y]], axis=1)])

        if update_alpha:
            self.update_alpha()

        self.Y_ref = np.vstack((self.Y_ref[1:], test_first))

    def update_alpha(self):
        c = sum([self.alpha[i] * self.calculate_K_sigma(self.Y_test[-1], self.Y_test[i]) for i in range(self.n_test)])

        self.alpha = np.append(self.alpha[1:] * (1 - self.eta * self.lambda_), self.eta / c)  # TODO: div by zero
        # self.alpha = np.append(self.alpha[1:] * (1 - self.eta * self.lambda_), c)

        b = self.calculate_b(self.Y_ref, self.Y_test)
        self.alpha += (1 - b.T * self.alpha) @ b / (b.T @ b)
        self.alpha = np.maximum(0, self.alpha)
        self.alpha /= b.T @ self.alpha

    def calculate_S(self):
        return sum([np.log(self.w_hat(self.Y_test[i])) for i in range(self.n_test)]) / self.n_test

    def kliep(self):
        if self.first_run:
            self.first_run = False
            self.w_hat = self.likelihood_cross_validation(self.Y_ref, self.Y_test)
        else:
            self.w_hat = self.fit(self.Y_ref, self.Y_test)

    def step(self, y):
        if not self.step_:
            self.update(y)
            S = self.calculate_S()

            if np.abs(S) > self.mu:
                self.step_ = True
                self.s_history.append(np.nan)
                # self.s_history.append(S)
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
        self.k = k  # TODO: <--
        self.t = 0
        size = n_test + n_ref + k
        self.s_history = [np.nan] * (size)
        self.t_list = np.arange(0, size, 1)

        def take():
            if Y.shape[0] >= self.t + size:
                slice_ = Y[self.t: self.t + size]
                self.t += size
                # TODO: np.lib.stride_tricks.sliding_window_view ?
                M = np.array([slice_[i:i + k, :] for i in range(self.n_ref + self.n_test)])
                # return slice_[:n_ref], slice_[n_ref:]
                return M[:self.n_ref], M[self.n_ref: self.n_ref + self.n_test]
            else:
                self.t += size
                return None

        def train():
            if (slice_ := take()) is not None:
                self.Y_ref, self.Y_test = slice_[0], slice_[1]
                self.kliep()
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


if __name__ == '__main__':
    # pass
    np.random.seed(42)


    def plot(k):
        fig, ax = plt.subplots(2, 1, figsize=(10, 6))
        ax[0].plot(k.Y)
        ax[1].plot(k.t_list, k.s_history)
        ax[1].plot(k.t_list, [k.mu] * len(k.t_list), linestyle='dashed', color='m')
        ax[1].plot(k.t_list, [-k.mu] * len(k.t_list), linestyle='dashed', color='m')
        ax[1].set_xlim(ax[0].get_xlim())
        ax[1].vlines(k.change_points, np.nanmin(k.s_history), np.nanmax(k.s_history), color='red')
        ax[0].vlines(k.change_points, np.nanmin(y), np.nanmax(y), color='red')
        plt.savefig('/tmp/fig.png')
        # plt.show()  # k.fit(y_ref, y_test, 1)  # k.lcv(y_ref, y_test, [10 ** i for i in range(3)])


    size = (200, 1)
    y = np.concatenate((
        np.random.normal(scale=5, size=size),
        np.random.normal(scale=25, size=size),
        np.random.normal(scale=5, size=size),
        np.random.normal(scale=25, size=size),
        # np.random.normal(loc=20, scale=5, size=size),
        # np.random.normal(loc=40, scale=5, size=size),
        # np.random.normal(loc=20, scale=5, size=size),
        # np.random.normal(scale=5, size=size),
    ))
    y = np.flip(y)

    n = 20
    n_ref = n_test = n
    k = n
    eta = 0.3
    lambda_ = .1
    mu = 2
    sigmas = [10, 25, 50, 100, 150, 200, 250]

    kliep = KLIEP(sigmas=sigmas, eta=eta, mu=mu, lambda_=lambda_, converge_iter=10000)
    kliep.run(y, n_ref, n_test, k=n)
    plot(kliep)
    print(kliep.change_points, kliep.sigma)
