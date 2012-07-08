from _pydnd import w_dtype as dtype, w_ndarray as ndarray, \
        make_byteswap_dtype, make_bytes_dtype, make_convert_dtype, \
        make_unaligned_dtype, make_fixedstring_dtype, make_string_dtype, \
        w_unary_gfunc as unary_gfunc, \
        arange, linspace

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
