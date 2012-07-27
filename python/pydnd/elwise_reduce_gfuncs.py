__all__ = []

import gfunc
import elwise_kernels

def add_basic_gfunc_from_elwise(root, elwise_root, types,
                associative, commutative, identity=None):
    global __all__
    f = gfunc.elwise_reduce(root)
    globals()[root] = f
    __all__.append(root)
    for t in types:
        name = elwise_root + '_' + t
        f.add_kernel(elwise_kernels.__dict__[name],
                associative=associative, commutative=commutative,
                identity=identity)

types = ['int32', 'int64', 'uint32', 'uint64', 'float32', 'float64']

add_basic_gfunc_from_elwise('sum', 'add', types,
        associative=True, commutative=True, identity=0)
add_basic_gfunc_from_elwise('product', 'multiply', types,
        associative=True, commutative=True, identity=1)
add_basic_gfunc_from_elwise('max', 'maximum2', types,
        associative=True, commutative=True)
add_basic_gfunc_from_elwise('min', 'minimum2', types,
        associative=True, commutative=True)
