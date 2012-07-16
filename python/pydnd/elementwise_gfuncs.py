__all__ = ['abs', 'floor', 'ceil', 'fmod', 'pow',
            'sqrt', 'exp', 'log', 'log10',
            'sin', 'cos', 'tan',
            'arcsin', 'arccos', 'arctan', 'arctan2',
            'sinh', 'cosh', 'tanh',
            'ldexp', 'isnan', 'isfinite', 'nextafter']

import gfunc
import elementwise_kernels

abs = gfunc.elwise('abs')
abs.add_kernel(elementwise_kernels.abs)
abs.add_kernel(elementwise_kernels.fabs)

floor = gfunc.elwise('floor')
floor.add_kernel(elementwise_kernels.floor)

ceil = gfunc.elwise('ceil')
ceil.add_kernel(elementwise_kernels.ceil)

fmod = gfunc.elwise('fmod')
fmod.add_kernel(elementwise_kernels.fmod)

pow = gfunc.elwise('pow')
pow.add_kernel(elementwise_kernels.pow)

sqrt = gfunc.elwise('sqrt')
sqrt.add_kernel(elementwise_kernels.sqrt)

exp = gfunc.elwise('exp')
exp.add_kernel(elementwise_kernels.exp)

log = gfunc.elwise('log')
log.add_kernel(elementwise_kernels.log)

log10 = gfunc.elwise('log10')
log10.add_kernel(elementwise_kernels.log10)

sin = gfunc.elwise('sin')
sin.add_kernel(elementwise_kernels.sin)

cos = gfunc.elwise('cos')
cos.add_kernel(elementwise_kernels.cos)

tan = gfunc.elwise('tan')
tan.add_kernel(elementwise_kernels.tan)

arcsin = gfunc.elwise('arcsin')
arcsin.add_kernel(elementwise_kernels.arcsin)

arccos = gfunc.elwise('arccos')
arccos.add_kernel(elementwise_kernels.arccos)

arctan = gfunc.elwise('arctan')
arctan.add_kernel(elementwise_kernels.arctan)

arctan2 = gfunc.elwise('arctan2')
arctan2.add_kernel(elementwise_kernels.arctan2)

sinh = gfunc.elwise('sinh')
sinh.add_kernel(elementwise_kernels.sinh)

cosh = gfunc.elwise('cosh')
cosh.add_kernel(elementwise_kernels.cosh)

tanh = gfunc.elwise('tanh')
tanh.add_kernel(elementwise_kernels.tanh)

ldexp = gfunc.elwise('ldexp')
ldexp.add_kernel(elementwise_kernels.ldexp)

isnan = gfunc.elwise('isnan')
isnan.add_kernel(elementwise_kernels.isnan)

isfinite = gfunc.elwise('isfinite')
isfinite.add_kernel(elementwise_kernels.isfinite)

nextafter = gfunc.elwise('nextafter')
nextafter.add_kernel(elementwise_kernels.nextafter)


