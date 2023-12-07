from mimetypes import init
import unittest
import numpy as np
from tests.test_base_mixin import TestBaseLinearModel, TestBaseLogisticModel
from src.algorithms import PLA
from src.models import LinearModel
from src.optimizers import SteepestDescentMethod, NewtonsMethod
from src.analyzers import PlotterAlgorithmObserver
import abc

class TestPLA(abc.ABC):
    pass

class TestPLARegression(TestBaseLinearModel, TestPLA):
        

    def test_steepest_descent_method(self):
        learning_rate = 0.1
        self.optimizer = SteepestDescentMethod(learning_rate)
        self.alg = PLA(self.optimizer, self.model)
        self.alg.fit(self.X, self.y, self.stop_criteria)
        expected = 0.974
        actual = self.alg.error
        self.assertAlmostEqual(expected, actual, delta=0.1)

    def test_steepest_newton_method(self):
        learning_rate = 1
        self.optimizer = NewtonsMethod(learning_rate)
        self.alg = PLA(self.optimizer, self.model)
        self.alg.fit(self.X, self.y, self.stop_criteria)
        expected = 0.994
        actual = self.alg.error
        self.assertAlmostEqual(expected, actual, delta=0.1)
    

if __name__ == '__main__':
    unittest.main()


class TestPLAClassification(TestBaseLogisticModel, TestPLA):

    def __init__(self, methodName: str = ...) -> None:
        super().__init__(methodName)
        
    def setUp(self) -> None:
        super().setUp()
   
    def test_steepest_descent_method(self):
        learning_rate = 0.0001 
        self.optimizer = SteepestDescentMethod(learning_rate)
        self.alg = PLA(self.optimizer, self.model)
        self.alg.fit(self.X, self.y, self.stop_criteria)
        expected = 16.13
        actual = self.alg.error
        self.assertAlmostEqual(expected, actual, delta=0.1)

    def test_steepest_newton_method(self):
        learning_rate = 0.1
        self.optimizer = NewtonsMethod(learning_rate)
        self.alg = PLA(self.optimizer, self.model)
        self.alg.fit(self.X, self.y, self.stop_criteria)
        expected = 10.1575
        actual = self.alg.error
        self.assertAlmostEqual(expected, actual, delta=0.1)
    

if __name__ == '__main__':
    unittest.main()