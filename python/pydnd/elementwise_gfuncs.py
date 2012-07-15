__all__ = ['abs', 'floor', 'ceil', 'sqrt', 'exp', 'log', 'log10',
            'sin', 'cos', 'tan',
            'arcsin', 'arccos', 'arctan',
            'sinh', 'cosh', 'tanh',
            'isnan', 'isfinite']

from _pydnd import w_elementwise_gfunc as elementwise_gfunc

import elementwise_kernels
from kernels import cgcache

abs = elementwise_gfunc('abs')
abs.add_kernel(cgcache, elementwise_kernels.abs)
abs.add_kernel(cgcache, elementwise_kernels.fabs)

floor = elementwise_gfunc('floor')
floor.add_kernel(cgcache, elementwise_kernels.floor)

ceil = elementwise_gfunc('ceil')
ceil.add_kernel(cgcache, elementwise_kernels.ceil)

sqrt = elementwise_gfunc('sqrt')
sqrt.add_kernel(cgcache, elementwise_kernels.sqrt)

exp = elementwise_gfunc('exp')
exp.add_kernel(cgcache, elementwise_kernels.exp)

log = elementwise_gfunc('log')
log.add_kernel(cgcache, elementwise_kernels.log)

log10 = elementwise_gfunc('log10')
log10.add_kernel(cgcache, elementwise_kernels.log10)

sin = elementwise_gfunc('sin')
sin.add_kernel(cgcache, elementwise_kernels.sin)

cos = elementwise_gfunc('cos')
cos.add_kernel(cgcache, elementwise_kernels.cos)

tan = elementwise_gfunc('tan')
tan.add_kernel(cgcache, elementwise_kernels.tan)

arcsin = elementwise_gfunc('arcsin')
arcsin.add_kernel(cgcache, elementwise_kernels.arcsin)

arccos = elementwise_gfunc('arccos')
arccos.add_kernel(cgcache, elementwise_kernels.arccos)

arctan = elementwise_gfunc('arctan')
arctan.add_kernel(cgcache, elementwise_kernels.arctan)

sinh = elementwise_gfunc('sinh')
sinh.add_kernel(cgcache, elementwise_kernels.sinh)

cosh = elementwise_gfunc('cosh')
cosh.add_kernel(cgcache, elementwise_kernels.cosh)

tanh = elementwise_gfunc('tanh')
tanh.add_kernel(cgcache, elementwise_kernels.tanh)

isnan = elementwise_gfunc('isnan')
isnan.add_kernel(cgcache, elementwise_kernels.isnan)

isfinite = elementwise_gfunc('isfinite')
isfinite.add_kernel(cgcache, elementwise_kernels.isfinite)


