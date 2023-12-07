import unittest
import numpy as np
from tests.test_base_mixin import TestBaseLinearModel, TestBaseLogisticModel
import abc

class TestModel(abc.ABC):
    pass

class TestLinearModel(TestBaseLinearModel, TestModel):

    def test_model_zero_weights(self):
        self.model.w = np.array([[0,0]]).T
        actual = self.model.error(self.X, self.y)
        expected = 1.0/len(self.X) * np.square(np.linalg.norm(self.y))     
        self.assertAlmostEqual(expected, actual, delta=0.1)

    def test_model_perfect_weights(self):
        self.model.w = self.weights
        actual = self.model.error(self.X, self.y)
        expected = 1
        self.assertAlmostEqual(expected, actual, delta=0.1)

class TestLogisticModel(TestBaseLogisticModel, TestModel):

    def test_model_zero_weights(self):
        self.model.w = np.array([[0,0]]).T
        actual = self.model.error(self.X, self.y)
        expected = 15.94
        self.assertAlmostEqual(expected, actual, delta=0.1)

    def test_model_perfect_weights(self):
        self.model.w = np.array([[15.0429, -.232163]]).T
        actual = self.model.error(self.X, self.y)
        expected = 10.15
        self.assertAlmostEqual(expected, actual, delta=0.1)


if __name__ == '__main__':
    unittest.main()