from _pydnd import w_dtype as dtype, w_ndarray as ndarray, \
        make_byteswap_dtype, make_fixedbytes_dtype, make_convert_dtype, \
        make_unaligned_dtype, make_fixedstring_dtype, make_string_dtype, \
        w_unary_gfunc as unary_gfunc, \
        arange, linspace
import dnd_ctypes as ctypes, elementwise_kernels

bool = dtype('bool')
int8 = dtype('int8')
int16 = dtype('int16')
int32 = dtype('int32')
int64 = dtype('int64')
uint8 = dtype('uint8')
uint16 = dtype('uint16')
uint32 = dtype('uint32')
uint64 = dtype('uint64')
float32 = dtype('float32')
float64 = dtype('float64')
cfloat32 = dtype('complex<float32>')
cfloat64 = dtype('complex<float64>')

abs = unary_gfunc('abs')
abs.add_kernel(elementwise_kernels.abs)
abs.add_kernel(elementwise_kernels.fabs)
#abs.add_kernel(elementwise_kernels.cabs)

floor = unary_gfunc('floor')
floor.add_kernel(elementwise_kernels.floor)

ceil = unary_gfunc('ceil')
ceil.add_kernel(elementwise_kernels.ceil)

sqrt = unary_gfunc('sqrt')
sqrt.add_kernel(elementwise_kernels.sqrt)

exp = unary_gfunc('exp')
exp.add_kernel(elementwise_kernels.exp)

log = unary_gfunc('log')
log.add_kernel(elementwise_kernels.log)

log10 = unary_gfunc('log10')
log10.add_kernel(elementwise_kernels.log10)

sin = unary_gfunc('sin')
sin.add_kernel(elementwise_kernels.sin)

cos = unary_gfunc('cos')
cos.add_kernel(elementwise_kernels.cos)

sinh = unary_gfunc('sinh')
sinh.add_kernel(elementwise_kernels.sinh)

cosh = unary_gfunc('cosh')
cosh.add_kernel(elementwise_kernels.cosh)


