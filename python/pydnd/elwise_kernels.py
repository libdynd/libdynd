import sys, os, ctypes, dnd_ctypes

if sys.platform == 'win32':
    libpath = os.path.join(os.path.dirname(__file__), "basic_kernels.dll")
    basic = ctypes.WinDLL(libpath)
else:
    libpath = os.path.join(os.path.dirname(__file__), "libbasic_kernels.so")
    basic = ctypes.CDLL(libpath)

def add_basic_kernels(root, types, nargs):
    """Adds simple ctypes kernels to the module namespace dict."""
    for t in types:
        name = root + '_' + t[0]
        func = basic[name]
        func.restype = t[1]
        func.argtypes = [t[1]] * nargs
        globals()[name] = func

types = [('int32', ctypes.c_int32),
         ('int64', ctypes.c_int64),
         ('uint32', ctypes.c_uint32),
         ('uint64', ctypes.c_uint64),
         ('float32', ctypes.c_float),
         ('float64', ctypes.c_double)]

add_basic_kernels('add', types, 2)
add_basic_kernels('subtract', types, 2)
add_basic_kernels('multiply', types, 2)
add_basic_kernels('divide', types, 2)
add_basic_kernels('maximum2', types, 2)
add_basic_kernels('minimum2', types, 2)
add_basic_kernels('maximum3', types, 3)
add_basic_kernels('minimum3', types, 3)
add_basic_kernels('square', types, 1)

if sys.platform == 'win32':
    # integer absolute value
    abs = ctypes.cdll.msvcrt.abs
    abs.restype = ctypes.c_int
    abs.argtypes = [ctypes.c_int]

    # float absolute value
    fabs = ctypes.cdll.msvcrt.fabs
    fabs.restype = ctypes.c_double
    fabs.argtypes = [ctypes.c_double]

    # floor
    floor = ctypes.cdll.msvcrt.floor
    floor.restype = ctypes.c_double
    floor.argtypes = [ctypes.c_double]

    # ceil
    ceil = ctypes.cdll.msvcrt.ceil
    ceil.restype = ctypes.c_double
    ceil.argtypes = [ctypes.c_double]

    # fmod
    fmod = ctypes.cdll.msvcrt.fmod
    fmod.restype = ctypes.c_double
    fmod.argtypes = [ctypes.c_double, ctypes.c_double]

    # pow
    pow = ctypes.cdll.msvcrt.pow
    pow.restype = ctypes.c_double
    pow.argtypes = [ctypes.c_double, ctypes.c_double]

    # square root
    sqrt = ctypes.cdll.msvcrt.sqrt
    sqrt.restype = ctypes.c_double
    sqrt.argtypes = [ctypes.c_double]

    # exp
    exp = ctypes.cdll.msvcrt.exp
    exp.restype = ctypes.c_double
    exp.argtypes = [ctypes.c_double]

    # log
    log = ctypes.cdll.msvcrt.log
    log.restype = ctypes.c_double
    log.argtypes = [ctypes.c_double]

    # log10
    log10 = ctypes.cdll.msvcrt.log10
    log10.restype = ctypes.c_double
    log10.argtypes = [ctypes.c_double]

    # sine
    sin = ctypes.cdll.msvcrt.sin
    sin.restype = ctypes.c_double
    sin.argtypes = [ctypes.c_double]

    # cosine
    cos = ctypes.cdll.msvcrt.cos
    cos.restype = ctypes.c_double
    cos.argtypes = [ctypes.c_double]

    # tangent
    tan = ctypes.cdll.msvcrt.tan
    tan.restype = ctypes.c_double
    tan.argtypes = [ctypes.c_double]

    # arc sine
    arcsin = ctypes.cdll.msvcrt.asin
    arcsin.restype = ctypes.c_double
    arcsin.argtypes = [ctypes.c_double]

    # arc cosine
    arccos = ctypes.cdll.msvcrt.acos
    arccos.restype = ctypes.c_double
    arccos.argtypes = [ctypes.c_double]

    # arc tan
    arctan = ctypes.cdll.msvcrt.atan
    arctan.restype = ctypes.c_double
    arctan.argtypes = [ctypes.c_double]

    # arc tan2
    arctan2 = ctypes.cdll.msvcrt.atan2
    arctan2.restype = ctypes.c_double
    arctan2.argtypes = [ctypes.c_double, ctypes.c_double]

    # hyperbolic sine
    sinh = ctypes.cdll.msvcrt.sinh
    sinh.restype = ctypes.c_double
    sinh.argtypes = [ctypes.c_double]

    # hyperbolic cosine
    cosh = ctypes.cdll.msvcrt.cosh
    cosh.restype = ctypes.c_double
    cosh.argtypes = [ctypes.c_double]

    # hyperbolic tangent
    tanh = ctypes.cdll.msvcrt.tanh
    tanh.restype = ctypes.c_double
    tanh.argtypes = [ctypes.c_double]

    # ldexp
    ldexp = ctypes.cdll.msvcrt.ldexp
    ldexp.restype = ctypes.c_double
    ldexp.argtypes = [ctypes.c_double, ctypes.c_int]

    # isnan
    isnan = ctypes.cdll.msvcrt._isnan
    isnan.restype = dnd_ctypes.c_dnd_bool
    isnan.argtypes = [ctypes.c_double]

    # isfinite
    isfinite = ctypes.cdll.msvcrt._finite
    isfinite.restype = dnd_ctypes.c_dnd_bool
    isfinite.argtypes = [ctypes.c_double]

    # nextafter
    nextafter = ctypes.cdll.msvcrt._nextafter
    nextafter.restype = ctypes.c_double
    nextafter.argtypes = [ctypes.c_double, ctypes.c_double]

# TODO: Add kernels for other platforms
