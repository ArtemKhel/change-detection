import enum
from abc import ABC, abstractmethod

import math
import numpy as np


# TODO:

# class Noise():
#     class Type(enum.Enum):
#         uni = 0
#         exp = 1
#         cauchy = 2
#         poisson = 3
#
#     def __init__(self, type: list[Type]):
#         pass
#
#
# class Wave(ABC):
#     def __init__(self, total_samples, n_features, f_height, f_size=None):
#         self._total_samples = total_samples
#         self._height = f_height
#         self.n_features = n_features
#         self._feature_size = f_size if f_size is not None else total_samples // n_features
#
#     def create(self):
#         self.wave = self.noise()
#         for i in range(self.n_features):
#             self.wave += self.feature(start=self._feature_size * i, end=self._feature_size * (i + 1))
#         return self.wave
#
#     @abstractmethod
#     def feature(self, start, end):
#         pass
#
#     def noise(self, amplitude=1.0):
#         return np.random.randn(self._total_samples, 1) * amplitude
#
#
# class VaryingAmplitudeNoise(Wave):
#     # Type = enum.Enum('Type', ['uni', 'exp', 'cauchy', 'poisson'])
#
#     def __init__(self, total_samples, n_features, height, type_=Noise.Type.uni):
#         self.type = type_
#         super().__init__(total_samples, n_features, height)
#
#     def feature(self, start, end):
#         def get():
#             match self.type:
#                 case Noise.Type.uni:
#                     return np.random.randn(math.ceil((end - start) / 2)) * self._height
#                 case Noise.Type.exp:
#                     return np.random.exponential(scale=self._height, size=math.ceil((end - start) / 2))
#                 case Noise.Type.cauchy:
#                     return np.random.standard_cauchy(math.ceil((end - start) / 2))
#                 case Noise.Type.poisson:
#                     return np.random.poisson(self._height, size=math.ceil((end - start) / 2))
#
#         noise = get()
#         return np.concatenate((
#             np.zeros(math.floor((start + (end - start) / 2))),
#             noise,
#             np.zeros((self._total_samples - end))
#         )).reshape((self._total_samples, 1))
#
#
# class Square_Wave(Wave):
#     def __init__(self, total_samples, n_features, height):
#         super().__init__(total_samples, n_features, height)
#
#     def feature(self, start, end):
#         return np.array([[self._height if start + (self._feature_size // 2) <= i < end else 0] for i in
#                          range(self._total_samples)])
#
#
# class Saw_Wave(Wave):
#     def __init__(self, total_samples, n_features, height):
#         super().__init__(total_samples, n_features, height)
#
#     def feature(self, start, end):
#         return np.concatenate((
#             np.zeros((start,)),
#             np.linspace(0, self._height, self._feature_size, dtype=int),
#             np.zeros((self._total_samples - end)))
#         ).reshape((self._total_samples, 1))
#
#
# class Triangle_Wave(Wave):
#     def __init__(self, total_samples, n_features, height):
#         super().__init__(total_samples, n_features, height)
#
#     def feature(self, start, end):
#         return np.concatenate((
#             np.zeros((start,)),
#             np.linspace(0, self._height, self._feature_size // 2, dtype=int),
#             np.linspace(self._height, 0, self._feature_size // 2, dtype=int),
#             np.zeros((self._total_samples - end)))
#         ).reshape((self._total_samples, 1))
