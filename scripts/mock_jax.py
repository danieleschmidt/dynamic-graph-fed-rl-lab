"""Mock JAX module for testing without JAX dependency."""

import numpy as np


class MockJNP:
    """Mock jax.numpy module."""
    
    def __getattr__(self, name):
        if hasattr(np, name):
            return getattr(np, name)
        return lambda *args, **kwargs: np.array([1.0, 0.0, 0.0, 0.0])
    
    def array(self, *args, **kwargs):
        return np.array(*args, **kwargs)
    
    def zeros(self, *args, **kwargs):
        return np.zeros(*args, **kwargs)
    
    def ones(self, *args, **kwargs):
        return np.ones(*args, **kwargs)
    
    def dot(self, *args, **kwargs):
        return np.dot(*args, **kwargs)
    
    def exp(self, *args, **kwargs):
        return np.exp(*args, **kwargs)
    
    def sqrt(self, *args, **kwargs):
        return np.sqrt(*args, **kwargs)
    
    def sum(self, *args, **kwargs):
        return np.sum(*args, **kwargs)
    
    def abs(self, *args, **kwargs):
        return np.abs(*args, **kwargs)
    
    def maximum(self, *args, **kwargs):
        return np.maximum(*args, **kwargs)
    
    def angle(self, *args, **kwargs):
        return np.angle(*args, **kwargs)
    
    def mean(self, *args, **kwargs):
        return np.mean(*args, **kwargs)
    
    def var(self, *args, **kwargs):
        return np.var(*args, **kwargs)
    
    complex64 = np.complex64
    float32 = np.float32


class MockJAX:
    """Mock JAX module."""
    
    def __init__(self):
        self.numpy = MockJNP()
    
    def jit(self, func):
        """Mock JIT decorator."""
        return func
    
    def vmap(self, func):
        """Mock vmap decorator.""" 
        return func
    
    def pmap(self, func):
        """Mock pmap decorator."""
        return func
    
    def grad(self, func):
        """Mock grad function."""
        return lambda x: x
    
    class config:
        @staticmethod
        def update(key, value):
            pass
    
    class random:
        @staticmethod
        def PRNGKey(seed):
            np.random.seed(seed % 2**32)
            return seed
        
        @staticmethod
        def choice(key, length, p=None):
            return np.random.choice(length, p=p)
        
        @staticmethod
        def uniform(key, minval=0.0, maxval=1.0):
            return np.random.uniform(minval, maxval)
        
        @staticmethod
        def normal(key):
            return np.random.normal()
        
        @staticmethod
        def permutation(key, arr):
            return np.random.permutation(arr)
        
        @staticmethod
        def split(key, num=2):
            return [key + i for i in range(num)]


# Install mock modules
import sys
sys.modules['jax'] = MockJAX()
sys.modules['jax.numpy'] = MockJNP()