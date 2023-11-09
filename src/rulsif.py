import logging

import numpy as np

from src.cpd_algorithm import CPD_Algorithm

logger = logging.getLogger(__name__)


# noinspection PyPep8Naming,PyAttributeOutsideInit
class RULSIF(CPD_Algorithm):
    def __init__(self, mu, lambda_, alpha=0.0, sigma=10):
        super().__init__(mu)
        self.sigma = sigma
        self.rulsif_alpha = alpha
        self.lambda_ = lambda_

        self.alpha = np.nan

    def fit(self):
        n_test = self.Y_test.shape[0]

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
        )[0]
