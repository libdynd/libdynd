__all__ = ['abs', 'floor', 'ceil', 'sqrt', 'exp', 'log', 'log10',
            'sin', 'cos', 'tan',
            'arcsin', 'arccos', 'arctan',
            'sinh', 'cosh', 'tanh',
            'isnan', 'isfinite']

from _pydnd import w_unary_gfunc as unary_gfunc

import elementwise_kernels
from kernels import cgcache

abs = unary_gfunc('abs')
abs.add_kernel(cgcache, elementwise_kernels.abs)
abs.add_kernel(cgcache, elementwise_kernels.fabs)
abs.add_kernel(cgcache, elementwise_kernels.cabs)

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

tan = unary_gfunc('tan')
tan.add_kernel(cgcache, elementwise_kernels.tan)

arcsin = unary_gfunc('arcsin')
arcsin.add_kernel(cgcache, elementwise_kernels.arcsin)

arccos = unary_gfunc('arccos')
arccos.add_kernel(cgcache, elementwise_kernels.arccos)

arctan = unary_gfunc('arctan')
arctan.add_kernel(cgcache, elementwise_kernels.arctan)

sinh = unary_gfunc('sinh')
sinh.add_kernel(cgcache, elementwise_kernels.sinh)

cosh = unary_gfunc('cosh')
cosh.add_kernel(cgcache, elementwise_kernels.cosh)

tanh = unary_gfunc('tanh')
tanh.add_kernel(cgcache, elementwise_kernels.tanh)

isnan = unary_gfunc('isnan')
isnan.add_kernel(cgcache, elementwise_kernels.isnan)

isfinite = unary_gfunc('isfinite')
isfinite.add_kernel(cgcache, elementwise_kernels.isfinite)


