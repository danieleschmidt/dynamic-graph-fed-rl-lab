"""Complete mock system for validation without external dependencies."""

import sys
import math
import random


class MockArray:
    """Mock array class to replace numpy arrays."""
    
    def __init__(self, data, dtype=None, shape=None):
        if isinstance(data, (list, tuple)):
            self.data = list(data)
            self.shape = (len(data),) if shape is None else shape
        elif isinstance(data, (int, float, complex)):
            self.data = [data]
            self.shape = (1,) if shape is None else shape
        else:
            self.data = [data]
            self.shape = (1,) if shape is None else shape
        
        self.dtype = dtype or type(self.data[0]) if self.data else float
        
    def __getitem__(self, index):
        return self.data[index]
    
    def __setitem__(self, index, value):
        self.data[index] = value
    
    def __len__(self):
        return len(self.data)
    
    def __add__(self, other):
        if isinstance(other, MockArray):
            return MockArray([a + b for a, b in zip(self.data, other.data)])
        return MockArray([x + other for x in self.data])
    
    def __mul__(self, other):
        if isinstance(other, MockArray):
            return MockArray([a * b for a, b in zip(self.data, other.data)])
        return MockArray([x * other for x in self.data])
    
    def __truediv__(self, other):
        if isinstance(other, MockArray):
            return MockArray([a / b for a, b in zip(self.data, other.data)])
        return MockArray([x / other for x in self.data])
    
    def sum(self):
        return sum(self.data)
    
    def mean(self):
        return sum(self.data) / len(self.data) if self.data else 0
    
    def squeeze(self, axis=None):
        return MockArray(self.data)


class MockNumPy:
    """Mock numpy module."""
    
    def __init__(self):
        self.complex64 = complex
        self.float32 = float
        self.pi = math.pi
        self.ndarray = MockArray
    
    def array(self, data, dtype=None):
        return MockArray(data, dtype)
    
    def zeros(self, shape, dtype=None):
        if isinstance(shape, int):
            size = shape
        else:
            size = shape[0] if shape else 1
        return MockArray([0] * size, dtype)
    
    def ones(self, shape, dtype=None):
        if isinstance(shape, int):
            size = shape
        else:
            size = shape[0] if shape else 1
        return MockArray([1] * size, dtype)
    
    def eye(self, n, dtype=None):
        data = []
        for i in range(n):
            row = [0] * n
            row[i] = 1
            data.extend(row)
        return MockArray(data, dtype, shape=(n, n))
    
    def dot(self, a, b):
        if hasattr(a, 'data') and hasattr(b, 'data'):
            return MockArray([sum(x * y for x, y in zip(a.data, b.data))])
        return MockArray([a * b])
    
    def exp(self, x):
        if hasattr(x, 'data'):
            return MockArray([math.exp(val) if isinstance(val, (int, float)) else val for val in x.data])
        return MockArray([math.exp(x)])
    
    def sqrt(self, x):
        if hasattr(x, 'data'):
            return MockArray([math.sqrt(val) if isinstance(val, (int, float)) else val for val in x.data])
        return MockArray([math.sqrt(x)])
    
    def abs(self, x):
        if hasattr(x, 'data'):
            return MockArray([abs(val) for val in x.data])
        return MockArray([abs(x)])
    
    def sum(self, x, axis=None):
        if hasattr(x, 'data'):
            return sum(x.data)
        return x
    
    def mean(self, x, axis=None):
        if hasattr(x, 'data'):
            return sum(x.data) / len(x.data) if x.data else 0
        return x
    
    def var(self, x, axis=None):
        if hasattr(x, 'data'):
            mean_val = self.mean(x)
            return sum((val - mean_val) ** 2 for val in x.data) / len(x.data)
        return 0
    
    def maximum(self, a, b):
        if hasattr(a, 'data') and hasattr(b, 'data'):
            return MockArray([max(x, y) for x, y in zip(a.data, b.data)])
        elif hasattr(a, 'data'):
            return MockArray([max(x, b) for x in a.data])
        elif hasattr(b, 'data'):
            return MockArray([max(a, x) for x in b.data])
        return MockArray([max(a, b)])
    
    def angle(self, x):
        if hasattr(x, 'data'):
            return MockArray([math.atan2(val.imag, val.real) if isinstance(val, complex) else 0 for val in x.data])
        return MockArray([math.atan2(x.imag, x.real) if isinstance(x, complex) else 0])
    
    def linspace(self, start, stop, num):
        if num <= 1:
            return MockArray([start])
        step = (stop - start) / (num - 1)
        return MockArray([start + i * step for i in range(num)])
    
    def allclose(self, a, b, rtol=1e-05, atol=1e-08):
        if hasattr(a, 'data') and hasattr(b, 'data'):
            return all(abs(x - y) <= (atol + rtol * abs(y)) for x, y in zip(a.data, b.data))
        elif hasattr(a, 'data'):
            return all(abs(x - b) <= (atol + rtol * abs(b)) for x in a.data)
        elif hasattr(b, 'data'):
            return all(abs(a - x) <= (atol + rtol * abs(x)) for x in b.data)
        return abs(a - b) <= (atol + rtol * abs(b))
    
    def stack(self, arrays):
        all_data = []
        for arr in arrays:
            if hasattr(arr, 'data'):
                all_data.extend(arr.data)
            else:
                all_data.append(arr)
        return MockArray(all_data)
    
    def concatenate(self, arrays, axis=None):
        all_data = []
        for arr in arrays:
            if hasattr(arr, 'data'):
                all_data.extend(arr.data)
            else:
                all_data.append(arr)
        return MockArray(all_data)


class MockJAXNumpy(MockNumPy):
    """Mock jax.numpy module extending numpy mock."""
    
    def __init__(self):
        super().__init__()
        self.ndarray = MockArray


class MockFlax:
    """Mock flax module."""
    
    class linen:
        class Module:
            def setup(self):
                pass
            
            def __call__(self, *args, **kwargs):
                return MockArray([1.0, 0.0, 0.0, 0.0])
        
        class Dense:
            def __init__(self, features):
                self.features = features
            
            def __call__(self, x):
                return MockArray([1.0] * self.features)
        
        class Sequential:
            def __init__(self, layers):
                self.layers = layers
            
            def __call__(self, x):
                return MockArray([1.0, 0.0])
        
        def softmax(x):
            if hasattr(x, 'data'):
                return MockArray([1.0 / len(x.data)] * len(x.data))
            return MockArray([1.0])
        
        def log_softmax(x):
            if hasattr(x, 'data'):
                return MockArray([0.0] * len(x.data))
            return MockArray([0.0])
        
        class activation:
            @staticmethod
            def relu(x):
                if hasattr(x, 'data'):
                    return MockArray([max(0, val) for val in x.data])
                return MockArray([max(0, x)])
            
            @staticmethod
            def tanh(x):
                if hasattr(x, 'data'):
                    return MockArray([math.tanh(val) for val in x.data])
                return MockArray([math.tanh(x)])
    
    class training:
        class TrainState:
            def __init__(self, step=0, apply_fn=None, params=None, tx=None, opt_state=None):
                self.step = step
                self.apply_fn = apply_fn
                self.params = params
                self.tx = tx
                self.opt_state = opt_state
            
            def apply_gradients(self, grads, **kwargs):
                return self
            
            @classmethod
            def create(cls, apply_fn, params, tx, **kwargs):
                return cls(apply_fn=apply_fn, params=params, tx=tx)


class MockOptax:
    """Mock optax module."""
    
    def adam(self, learning_rate):
        return self
    
    def sgd(self, learning_rate):
        return self
    
    def apply_updates(self, params, updates):
        return params
    
    def update(self, updates, state, params=None):
        return updates, state
    
    class Schedule:
        def __init__(self, value):
            self.value = value
        
        def __call__(self, step):
            return self.value
    
    class GradientTransformation:
        def __init__(self):
            pass
        
        def init(self, params):
            return {}
        
        def update(self, updates, state, params=None):
            return updates, state


class MockPsutil:
    """Mock psutil module."""
    
    class Process:
        def __init__(self, pid=None):
            pass
        
        def memory_info(self):
            class MemInfo:
                rss = 1024 * 1024
                vms = 2048 * 1024
            return MemInfo()
        
        def memory_percent(self):
            return 15.5


class MockGymnasium:
    """Mock gymnasium module."""
    
    class Env:
        def __init__(self):
            self.observation_space = self
            self.action_space = self
            self.shape = (4,)
        
        def reset(self):
            return MockArray([0.0, 0.0, 0.0, 0.0]), {}
        
        def step(self, action):
            return MockArray([0.0, 0.0, 0.0, 0.0]), 0.0, False, False, {}
        
        def close(self):
            pass
    
    @staticmethod
    def make(env_id):
        return MockGymnasium.Env()
    
    class spaces:
        class Space:
            def __init__(self):
                self.shape = (4,)
            
            def sample(self):
                return MockArray([0.5, 0.5, 0.5, 0.5])
        
        class Box(Space):
            def __init__(self, low, high, shape=None, dtype=None):
                super().__init__()
                self.low = low
                self.high = high
                self.shape = shape or (4,)
                self.dtype = dtype
            
            def sample(self):
                return MockArray([0.5] * self.shape[0])
        
        class Discrete(Space):
            def __init__(self, n):
                super().__init__()
                self.n = n
                self.shape = (1,)
            
            def sample(self):
                return 0
    
    Space = spaces.Space


class MockNetworkX:
    """Mock networkx module."""
    
    class Graph:
        def __init__(self):
            self.nodes_dict = {}
            self.edges_list = []
        
        def add_node(self, node, **attr):
            self.nodes_dict[node] = attr
        
        def add_edge(self, u, v, **attr):
            self.edges_list.append((u, v, attr))
        
        def nodes(self):
            return list(self.nodes_dict.keys())
        
        def edges(self):
            return [(u, v) for u, v, _ in self.edges_list]
        
        def number_of_nodes(self):
            return len(self.nodes_dict)
        
        def number_of_edges(self):
            return len(self.edges_list)
    
    @staticmethod
    def erdos_renyi_graph(n, p):
        g = MockNetworkX.Graph()
        for i in range(n):
            g.add_node(i)
        # Add some edges
        for i in range(min(n-1, 5)):  # Add up to 5 edges
            g.add_edge(i, (i+1) % n)
        return g
    
    @staticmethod
    def barabasi_albert_graph(n, m):
        g = MockNetworkX.Graph()
        for i in range(n):
            g.add_node(i)
        for i in range(min(n-1, m)):
            g.add_edge(i, (i+1) % n)
        return g


class MockJAX:
    """Mock JAX module."""
    
    def __init__(self):
        self.numpy = MockJAXNumpy()
    
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
            random.seed(seed % (2**32))
            return seed
        
        @staticmethod
        def choice(key, length, p=None):
            return random.randint(0, length - 1)
        
        @staticmethod
        def uniform(key, minval=0.0, maxval=1.0):
            return random.uniform(minval, maxval)
        
        @staticmethod
        def normal(key):
            return random.gauss(0, 1)
        
        @staticmethod
        def permutation(key, arr):
            if hasattr(arr, 'data'):
                data = arr.data.copy()
                random.shuffle(data)
                return MockArray(data)
            return arr
        
        @staticmethod
        def split(key, num=2):
            return [key + i for i in range(num)]
        
        @staticmethod
        def fold_in(key, data):
            return key + hash(str(data)) % 1000


# Install all mocks
def install_mocks():
    """Install all mock modules."""
    numpy_mock = MockNumPy()
    jax_mock = MockJAX()
    optax_mock = MockOptax()
    flax_mock = MockFlax()
    psutil_mock = MockPsutil()
    gymnasium_mock = MockGymnasium()
    networkx_mock = MockNetworkX()
    
    sys.modules['numpy'] = numpy_mock
    sys.modules['np'] = numpy_mock
    sys.modules['jax'] = jax_mock
    sys.modules['jax.numpy'] = jax_mock.numpy
    sys.modules['jnp'] = jax_mock.numpy
    sys.modules['optax'] = optax_mock
    sys.modules['flax'] = flax_mock
    sys.modules['flax.linen'] = flax_mock.linen
    sys.modules['flax.training'] = flax_mock.training
    sys.modules['flax.training.train_state'] = flax_mock.training
    sys.modules['psutil'] = psutil_mock
    sys.modules['gymnasium'] = gymnasium_mock
    sys.modules['gym'] = gymnasium_mock  # Legacy gym
    sys.modules['networkx'] = networkx_mock
    sys.modules['nx'] = networkx_mock
    
    return numpy_mock, jax_mock, optax_mock, flax_mock, psutil_mock, gymnasium_mock, networkx_mock


if __name__ == "__main__":
    install_mocks()
    print("All mocks installed successfully")