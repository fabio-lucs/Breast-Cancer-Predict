from __future__ import annotations
import numpy as np
import scipy.stats as st
import abc
import src.models as models

class OptimizerStrategy(abc.ABC):
    def __init__(self, learning_rate: float) -> None:
        self.learning_rate = learning_rate
    
    @abc.abstractmethod
    def update_model(self, X, y, model: models.Model):
        """Implement Update Weigth Strategy"""
    
class NewtonsMethod(OptimizerStrategy):

    def update_model(self, X, y, model: models.Model):
        gradient = model.gradient(X , y)
        
        hessian = model.hessian(X, y)
        hessian += self.learning_rate * np.eye(hessian.shape[0])
        hessian_inv = np.linalg.inv(hessian)
        
        model.w = model.w  - self.learning_rate * np.dot(hessian_inv, gradient)
        

class SteepestDescentMethod(OptimizerStrategy):

    def update_model(self, X, y, model: models.Model):
        gradient = model.gradient(X, y)
        model.w = model.w - self.learning_rate * gradient



