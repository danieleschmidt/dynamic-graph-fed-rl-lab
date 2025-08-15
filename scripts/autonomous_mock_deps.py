#!/usr/bin/env python3
"""
Autonomous mock dependencies for Generation 1 execution.
Creates lightweight mocks for JAX, PyTorch, and other dependencies.
"""

import sys
import os
from unittest.mock import MagicMock, Mock
from types import ModuleType

# Mock JAX ecosystem
def create_jax_mock():
    jax = ModuleType('jax')
    jax.numpy = ModuleType('jax.numpy')
    jax.random = ModuleType('jax.random')
    
    # JAX numpy mock with ndarray
    class MockNdarray:
        def __init__(self, data):
            self.data = data
            self.shape = (len(data),) if hasattr(data, '__iter__') else (1,)
            
        def __getitem__(self, key):
            return self.data[key] if hasattr(self.data, '__getitem__') else self.data
            
        def __len__(self):
            return len(self.data) if hasattr(self.data, '__len__') else 1
    
    jax.numpy.ndarray = MockNdarray
    jax.numpy.array = lambda x: MockNdarray(x)
    jax.numpy.zeros = lambda *args, **kwargs: MockNdarray([0] * (args[0] if args else 1))
    jax.numpy.ones = lambda *args, **kwargs: MockNdarray([1] * (args[0] if args else 1))
    jax.numpy.sum = lambda x, **kwargs: sum(x.data) if hasattr(x, 'data') and hasattr(x.data, '__iter__') else x
    jax.numpy.mean = lambda x, **kwargs: sum(x.data) / len(x.data) if hasattr(x, 'data') and hasattr(x.data, '__iter__') and len(x.data) > 0 else 0
    jax.numpy.sqrt = lambda x: x ** 0.5
    jax.numpy.exp = lambda x: 2.718 ** x if isinstance(x, (int, float)) else [2.718 ** i for i in x]
    
    # JAX random mock
    jax.random.PRNGKey = lambda x: x
    jax.random.choice = lambda key, length, p=None: 0 if length > 0 else None
    jax.random.normal = lambda key, shape: [0.1] * (shape[0] if hasattr(shape, '__iter__') else shape)
    jax.random.permutation = lambda key, x: x
    
    # JAX core functions
    jax.grad = lambda f: lambda *args: [0.1] * len(args)
    jax.jit = lambda f: f
    jax.vmap = lambda f: f
    
    return jax

def create_torch_mock():
    torch = ModuleType('torch')
    torch.nn = ModuleType('torch.nn')
    torch.optim = ModuleType('torch.optim')
    torch.nn.functional = ModuleType('torch.nn.functional')
    
    # Basic tensor mock
    class MockTensor:
        def __init__(self, data):
            self.data = data
            self.shape = (len(data),) if hasattr(data, '__iter__') else (1,)
        
        def mean(self, dim=None):
            if hasattr(self.data, '__iter__'):
                return sum(self.data) / len(self.data)
            return self.data
        
        def sum(self, dim=None):
            return sum(self.data) if hasattr(self.data, '__iter__') else self.data
            
        def backward(self):
            pass
            
        def detach(self):
            return self
    
    torch.tensor = lambda x: MockTensor(x)
    torch.zeros = lambda *args: MockTensor([0] * args[0] if args else [0])
    torch.ones = lambda *args: MockTensor([1] * args[0] if args else [1])
    torch.stack = lambda x: MockTensor(x)
    torch.cat = lambda x, dim=None: MockTensor(sum(x, []) if hasattr(x[0], '__iter__') else x)
    
    # Neural network mocks
    torch.nn.Module = type('Module', (), {'__init__': lambda self: None, 'forward': lambda self, x: x})
    torch.nn.Linear = lambda in_f, out_f: type('Linear', (torch.nn.Module,), {
        '__init__': lambda self: super().__init__(),
        'forward': lambda self, x: MockTensor([0.5] * out_f) if isinstance(x, MockTensor) else MockTensor([0.5])
    })()
    torch.nn.GRU = lambda **kwargs: type('GRU', (), {'forward': lambda self, x: (x, x)})()
    torch.nn.MultiheadAttention = lambda **kwargs: type('MultiheadAttention', (), {
        'forward': lambda self, x, y, z: (x, MockTensor([0.1]))
    })()
    
    # Functional operations
    torch.nn.functional.mse_loss = lambda x, y: 0.1
    torch.nn.functional.relu = lambda x: x
    
    # Optimizers
    torch.optim.Adam = lambda params, **kwargs: type('Adam', (), {
        'step': lambda self: None,
        'zero_grad': lambda self: None
    })()
    
    return torch

def create_other_mocks():
    """Create mocks for other dependencies."""
    mocks = {}
    
    # NetworkX mock
    networkx = ModuleType('networkx')
    networkx.Graph = lambda: type('Graph', (), {
        'add_node': lambda self, n: None,
        'add_nodes_from': lambda self, nodes: None,
        'add_edge': lambda self, u, v: None,
        'add_edges_from': lambda self, edges: None,
        'neighbors': lambda self, node: [],
        'nodes': lambda self: [],
        'edges': lambda self: []
    })()
    mocks['networkx'] = networkx
    
    # Gymnasium mock
    gymnasium = ModuleType('gymnasium')
    gymnasium.Env = type('Env', (), {
        'reset': lambda self: ([0], {}),
        'step': lambda self, action: ([0], 0, False, False, {}),
        'close': lambda self: None
    })
    gymnasium.Space = type('Space', (), {})
    gymnasium.spaces = ModuleType('gymnasium.spaces')
    gymnasium.spaces.Space = gymnasium.Space
    gymnasium.spaces.Box = lambda low, high, shape=None, dtype=None: gymnasium.Space
    gymnasium.spaces.Discrete = lambda n: gymnasium.Space
    gymnasium.spaces.Dict = lambda spaces: gymnasium.Space
    mocks['gymnasium'] = gymnasium
    
    # Also support 'gym' alias
    mocks['gym'] = gymnasium
    
    # Torch geometric mock
    torch_geometric = ModuleType('torch_geometric')
    torch_geometric.nn = ModuleType('torch_geometric.nn')
    torch_geometric.nn.GCNConv = lambda in_channels, out_channels: type('GCNConv', (), {
        'forward': lambda self, x, edge_index: x
    })()
    mocks['torch_geometric'] = torch_geometric
    
    # Numpy (enhanced mock)
    try:
        import numpy
        # Ensure ndarray is available
        if not hasattr(numpy, 'ndarray'):
            numpy.ndarray = type('ndarray', (), {})
        mocks['numpy'] = numpy
    except ImportError:
        numpy = ModuleType('numpy')
        numpy.ndarray = type('ndarray', (), {})
        numpy.array = lambda x: x
        numpy.zeros = lambda *args, **kwargs: [[0] * args[1] for _ in range(args[0])] if len(args) > 1 else [0] * args[0] if args else [0]
        numpy.ones = lambda *args, **kwargs: [1] * args[0] if args else [1]
        numpy.sqrt = lambda x: x ** 0.5
        numpy.exp = lambda x: 2.718 ** x if isinstance(x, (int, float)) else [2.718 ** i for i in x]
        numpy.sum = lambda x, **kwargs: sum(x) if hasattr(x, '__iter__') else x
        numpy.random = ModuleType('numpy.random')
        numpy.random.choice = lambda x, size=None, p=None: x[0] if hasattr(x, '__iter__') and len(x) > 0 else 0
        numpy.random.permutation = lambda key, x: x
        mocks['numpy'] = numpy
    
    # Ensure np alias  
    mocks['np'] = mocks['numpy']
    
    # Optax mock for JAX optimization
    optax = ModuleType('optax')
    optax.Schedule = type('Schedule', (), {})  # Mock schedule type
    optax.GradientTransformation = type('GradientTransformation', (), {})  # Mock gradient transformation
    optax.adam = lambda lr: type('AdamState', (), {
        'init': lambda self, params: {},
        'update': lambda self, grads, state, params: (grads, state)
    })()
    optax.apply_updates = lambda params, updates: params
    optax.linear_schedule = lambda init_value, end_value, transition_steps: optax.Schedule
    optax.cosine_decay_schedule = lambda init_value, decay_steps: optax.Schedule
    optax.chain = lambda *args: optax.GradientTransformation
    optax.clip_by_global_norm = lambda max_norm: optax.GradientTransformation
    mocks['optax'] = optax
    
    # Additional ML/AI mocks
    wandb = ModuleType('wandb')
    wandb.init = lambda **kwargs: None
    wandb.log = lambda **kwargs: None
    wandb.finish = lambda: None
    mocks['wandb'] = wandb
    
    # Ray mock
    ray = ModuleType('ray')
    ray.init = lambda **kwargs: None
    ray.shutdown = lambda: None
    mocks['ray'] = ray
    
    # Flax mock (JAX neural networks)
    flax = ModuleType('flax')
    flax.linen = ModuleType('flax.linen')
    flax.linen.Module = type('Module', (), {
        '__init__': lambda self: None,
        'setup': lambda self: None,
        '__call__': lambda self, x: x
    })
    flax.linen.Dense = lambda features, **kwargs: type('Dense', (flax.linen.Module,), {
        '__call__': lambda self, x: [0.5] * features
    })()
    flax.core = ModuleType('flax.core')
    flax.core.freeze = lambda x: x
    mocks['flax'] = flax
    
    # Additional scientific computing mocks
    scipy = ModuleType('scipy')
    scipy.sparse = ModuleType('scipy.sparse')
    scipy.sparse.csr_matrix = lambda x: x
    mocks['scipy'] = scipy
    
    # FastAPI and web framework mocks
    fastapi = ModuleType('fastapi')
    fastapi.FastAPI = lambda **kwargs: type('FastAPI', (), {
        'add_api_route': lambda self, *args, **kwargs: None,
        'mount': lambda self, *args, **kwargs: None
    })()
    fastapi.HTTPException = Exception
    mocks['fastapi'] = fastapi
    
    uvicorn = ModuleType('uvicorn')
    uvicorn.run = lambda app, **kwargs: None
    mocks['uvicorn'] = uvicorn
    
    # Pydantic mock
    pydantic = ModuleType('pydantic')
    pydantic.BaseModel = type('BaseModel', (), {})
    mocks['pydantic'] = pydantic
    
    # Quantum computing mocks
    qiskit = ModuleType('qiskit')
    qiskit.QuantumCircuit = lambda *args: type('QuantumCircuit', (), {
        'h': lambda self, qubit: None,
        'cx': lambda self, control, target: None,
        'measure_all': lambda self: None
    })()
    qiskit.execute = lambda circuit, backend: type('Job', (), {
        'result': lambda self: type('Result', (), {
            'get_counts': lambda self: {'00': 512, '11': 512}
        })()
    })()
    mocks['qiskit'] = qiskit
    
    cirq = ModuleType('cirq')
    cirq.GridQubit = lambda row, col: f"q_{row}_{col}"
    cirq.Circuit = lambda: type('Circuit', (), {
        'append': lambda self, gate: None
    })()
    mocks['cirq'] = cirq
    
    # psutil mock for system monitoring
    psutil = ModuleType('psutil')
    psutil.cpu_percent = lambda interval=None: 25.5 + (time.time() % 50)
    psutil.virtual_memory = lambda: type('Memory', (), {
        'percent': 45.2,
        'rss': 500 * 1024 * 1024,  # 500MB
        'available': 8192 * 1024 * 1024
    })()
    psutil.disk_usage = lambda path: type('Disk', (), {
        'used': 500 * 1024 * 1024 * 1024,
        'total': 1000 * 1024 * 1024 * 1024,
        'free': 500 * 1024 * 1024 * 1024
    })()
    psutil.net_io_counters = lambda: type('NetIO', (), {
        'bytes_sent': 1024000,
        'bytes_recv': 2048000,
        'packets_sent': 1000,
        'packets_recv': 1500,
        'read_bytes': 1024000,
        'write_bytes': 2048000
    })()
    psutil.disk_io_counters = lambda: type('DiskIO', (), {
        'read_bytes': 1024000,
        'write_bytes': 2048000
    })()
    psutil.Process = lambda: type('Process', (), {
        'memory_info': lambda self: type('MemInfo', (), {'rss': 100 * 1024 * 1024})()
    })()
    mocks['psutil'] = psutil
    
    return mocks

def setup_autonomous_mocks():
    """Setup all mocks for autonomous execution."""
    import time  # Import time for psutil mock
    
    # Install JAX mock
    sys.modules['jax'] = create_jax_mock()
    sys.modules['jax.numpy'] = sys.modules['jax'].numpy
    sys.modules['jax.random'] = sys.modules['jax'].random
    sys.modules['jaxlib'] = ModuleType('jaxlib')
    
    # Install PyTorch mock
    sys.modules['torch'] = create_torch_mock()
    sys.modules['torch.nn'] = sys.modules['torch'].nn
    sys.modules['torch.optim'] = sys.modules['torch'].optim
    sys.modules['torch.nn.functional'] = sys.modules['torch'].nn.functional
    
    # Install other mocks
    other_mocks = create_other_mocks()
    for name, mock in other_mocks.items():
        sys.modules[name] = mock
        # Also install submodules
        if name == 'flax':
            sys.modules['flax.linen'] = mock.linen
            sys.modules['flax.core'] = mock.core
            sys.modules['flax.training'] = ModuleType('flax.training')
            sys.modules['flax.training.train_state'] = ModuleType('flax.training.train_state')
            sys.modules['flax.training.train_state'].TrainState = type('TrainState', (), {
                'create': lambda **kwargs: type('TrainState', (), {
                    'apply_fn': lambda self, params, x: x,
                    'params': {},
                    'tx': None,
                    'opt_state': None
                })()
            })
            sys.modules['flax.training'].train_state = sys.modules['flax.training.train_state']
    
    print("✅ Autonomous mock dependencies installed successfully")
    print("🚀 Ready for Generation 1 execution")

if __name__ == "__main__":
    setup_autonomous_mocks()