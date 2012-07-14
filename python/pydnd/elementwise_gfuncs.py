__all__ = ['floor', 'ceil', 'sqrt', 'exp', 'log', 'log10',
            'sin', 'cos', 'sinh', 'cosh']

from _pydnd import w_unary_gfunc as unary_gfunc

import elementwise_kernels
from kernels import cgcache

abs = unary_gfunc('abs')
abs.add_kernel(cgcache, elementwise_kernels.abs)
abs.add_kernel(cgcache, elementwise_kernels.fabs)
#abs.add_kernel(cgcache, elementwise_kernels.cabs)

floor = unary_gfunc('floor')
floor.add_kernel(cgcache, elementwise_kernels.floor)

ceil = unary_gfunc('ceil')
ceil.add_kernel(cgcache, elementwise_kernels.ceil)

sqrt = unary_gfunc('sqrt')
sqrt.add_kernel(cgcache, elementwise_kernels.sqrt)

exp = unary_gfunc('exp')
exp.add_kernel(cgcache, elementwise_kernels.exp)

log = unary_gfunc('log')
log.add_kernel(cgcache, elementwise_kernels.log)

log10 = unary_gfunc('log10')
log10.add_kernel(cgcache, elementwise_kernels.log10)

sin = unary_gfunc('sin')
sin.add_kernel(cgcache, elementwise_kernels.sin)

cos = unary_gfunc('cos')
cos.add_kernel(cgcache, elementwise_kernels.cos)

sinh = unary_gfunc('sinh')
sinh.add_kernel(cgcache, elementwise_kernels.sinh)

cosh = unary_gfunc('cosh')
cosh.add_kernel(cgcache, elementwise_kernels.cosh)


