from __future__ import annotations
import numpy as np
import scipy.stats as st
import abc
import src.optimizers as opt



class Model(abc.ABC):

    def __init__(self) -> None:
        self._w = None
        super().__init__()
        
    
    @abc.abstractmethod
    def predict(self, X: np.array) -> np.ndarray:
        """Implement the predict method"""

    @abc.abstractmethod
    def gradient(self, X: np.array, y: np.array) -> np.ndarray:
        """Implement gradient"""
        
    @abc.abstractmethod
    def hessian(self, X: np.array, y: np.array) -> np.ndarray:
        """Implement hessian"""

    @abc.abstractmethod
    def error(self, X, y):
        """Implement error""" #  def __update_error(self, X, y):
        
        
    @property
    def w(self):
        return self._w

    @w.setter
    def w(self, value):
        self._w = value

class LinearModel(Model):

    def __init__(self) -> None:
        self._w = None
        super().__init__()
          
    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.dot(X, self.w)

    def gradient(self, X: np.array, y: np.array) -> np.ndarray:
        return (2/len(X)) * (np.linalg.multi_dot([X.T, X, self.w]) - np.dot(X.T,y))

    def hessian(self, X: np.array, y: np.array) -> np.ndarray:
        return (2/len(X))*np.dot(X.T, X)
            
    def error(self, X, y):
            yhat = self.predict(X)
            errors = yhat - y
            rmse = 1.0/len(X) * np.square(np.linalg.norm(errors))     
            return rmse
    
class LogisticModel(Model):

    def __init__(self) -> None:
        super().__init__()
        self._w = None

    def predict(self, X: np.array) -> np.ndarray:
       return self.sigmoid(X)

    def gradient(self, X: np.array, y: np.array) -> np.ndarray:
        p = self.sigmoid(X)
        return np.dot(X.T, (p - y.reshape(-1, 1)))

    def hessian(self, X: np.array, y: np.array) -> np.ndarray:
        p = self.sigmoid(X)
        P = np.diag(p.flatten() * (1 - p.flatten()))
        return np.dot(np.dot(X.T, P), X)
    
    def sigmoid(self, x):
        return  1 / (1 + np.exp(-np.dot(x,self._w)))

    def error(self, X, y):
         p = self.predict(X)
         return np.sum(-y * np.log(p) - (1 - y) * np.log(1 - p))
