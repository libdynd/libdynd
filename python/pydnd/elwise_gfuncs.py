__all__ = ['abs', 'floor', 'ceil', 'fmod', 'pow',
            'sqrt', 'exp', 'log', 'log10',
            'sin', 'cos', 'tan',
            'arcsin', 'arccos', 'arctan', 'arctan2',
            'sinh', 'cosh', 'tanh',
            'ldexp', 'isnan', 'isfinite', 'nextafter']

import gfunc
import elwise_kernels

def add_basic_gfunc(root, types, kernel_roots = None):
    if kernel_roots is None:
        kernel_roots = [root]
    global __all__
    f = gfunc.elwise(root)
    globals()[root] = f
    __all__.append(root)
    for r in kernel_roots:
        for t in types:
            name = r + '_' + t
            f.add_kernel(elwise_kernels.__dict__[name])

types = ['int32', 'int64', 'uint32', 'uint64', 'float32', 'float64']

add_basic_gfunc('add', types)
add_basic_gfunc('subtract', types)
add_basic_gfunc('multiply', types)
add_basic_gfunc('divide', types)
#add_basic_gfunc('maximum', types, ['maximum2', 'maximum3'])
#add_basic_gfunc('minimum', types, ['minimum2', 'minimum3'])
add_basic_gfunc('maximum', types, ['maximum2'])
add_basic_gfunc('minimum', types, ['minimum2'])
add_basic_gfunc('square', types)

abs = gfunc.elwise('abs')
abs.add_kernel(elwise_kernels.abs)
abs.add_kernel(elwise_kernels.fabs)

floor = gfunc.elwise('floor')
floor.add_kernel(elwise_kernels.floor)

ceil = gfunc.elwise('ceil')
ceil.add_kernel(elwise_kernels.ceil)

fmod = gfunc.elwise('fmod')
fmod.add_kernel(elwise_kernels.fmod)

pow = gfunc.elwise('pow')
pow.add_kernel(elwise_kernels.pow)

sqrt = gfunc.elwise('sqrt')
sqrt.add_kernel(elwise_kernels.sqrt)

exp = gfunc.elwise('exp')
exp.add_kernel(elwise_kernels.exp)

log = gfunc.elwise('log')
log.add_kernel(elwise_kernels.log)

log10 = gfunc.elwise('log10')
log10.add_kernel(elwise_kernels.log10)

sin = gfunc.elwise('sin')
sin.add_kernel(elwise_kernels.sin)

cos = gfunc.elwise('cos')
cos.add_kernel(elwise_kernels.cos)

tan = gfunc.elwise('tan')
tan.add_kernel(elwise_kernels.tan)

arcsin = gfunc.elwise('arcsin')
arcsin.add_kernel(elwise_kernels.arcsin)

arccos = gfunc.elwise('arccos')
arccos.add_kernel(elwise_kernels.arccos)

arctan = gfunc.elwise('arctan')
arctan.add_kernel(elwise_kernels.arctan)

arctan2 = gfunc.elwise('arctan2')
arctan2.add_kernel(elwise_kernels.arctan2)

sinh = gfunc.elwise('sinh')
sinh.add_kernel(elwise_kernels.sinh)

cosh = gfunc.elwise('cosh')
cosh.add_kernel(elwise_kernels.cosh)

tanh = gfunc.elwise('tanh')
tanh.add_kernel(elwise_kernels.tanh)

ldexp = gfunc.elwise('ldexp')
ldexp.add_kernel(elwise_kernels.ldexp)

isnan = gfunc.elwise('isnan')
isnan.add_kernel(elwise_kernels.isnan)

isfinite = gfunc.elwise('isfinite')
isfinite.add_kernel(elwise_kernels.isfinite)

nextafter = gfunc.elwise('nextafter')
nextafter.add_kernel(elwise_kernels.nextafter)


