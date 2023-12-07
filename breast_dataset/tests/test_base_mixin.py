from __future__ import annotations
import unittest
from abc import ABC, abstractmethod
import numpy as np
from src.data_generator import DataGeneratorLinearModel
from src.models import LinearModel, LogisticModel
from src.stop_criteria import CompositeStopCriteria, MaxIterationStopCriteria, MinErrorStopCriteria
from src.preprocessing import Preprocessing

class TestBaseMixin(unittest.TestCase, ABC):

    def __init__(self, methodName: str = ...) -> None:
        super().__init__(methodName)

    def setUp(self) -> None:
        super().setUp()
        np.random.seed(0)
        min_error = 1
        max_iteration = 100
        self.stop_criteria = CompositeStopCriteria()
        self.stop_criteria.add(MinErrorStopCriteria(min_error=min_error))
        self.stop_criteria.add(MaxIterationStopCriteria(max_iteration=max_iteration))
        
        
    @property
    def sample_size(self):
        return self._sample_size
    
    @property
    def X(self):
        return self._X

    @property
    def y(self):
        return self._y

class TestBaseLinearModel(TestBaseMixin):

    def __init__(self, methodName: str = ...) -> None:
        super().__init__(methodName)

    def setUp(self) -> None:
        super().setUp()        
        self.model = LinearModel()
        w0 = 10
        w1 = 4
        self._sample_size = 1000
        self.weights = np.array([[w0, w1]]).T
        self.x_min = -2
        self.x_max = 2
        self.generate_data()
        
        
    def generate_data(self):
        self.data_generator = DataGeneratorLinearModel(self.sample_size, self.weights, self.x_min, self.x_max )
        X, y = self.data_generator.get_data()
        self._X = X
        self._y = y

class TestBaseLogisticModel(TestBaseMixin):

    def setUp(self) -> None:
        super().setUp()        
        self.model = LogisticModel()
        data = np.loadtxt("data.csv", delimiter=",")
        X = Preprocessing.build_design_matrix(data[:, [0]])
        y = data[:, [1]]
        self._X = X
        self._y = y