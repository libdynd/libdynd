__all__ = ['c_complex_float32', 'c_complex64',
            'c_complex_float64', 'c_complex128']

import ctypes, _pydnd

class c_complex_float32(ctypes.Structure):
    _fields_ = [('real', ctypes.c_float),
                ('imag', ctypes.c_float)]
    _dynd_type_ = _pydnd.w_dtype('complex<float32>')

class c_complex_float64(ctypes.Structure):
    _fields_ = [('real', ctypes.c_double),
                ('imag', ctypes.c_double)]
    _dynd_type_ = _pydnd.w_dtype('complex<float64>')

c_complex64 = c_complex_float32
c_complex128 = c_complex_float64
