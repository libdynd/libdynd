import mpmath

from mpmath import pi, mpc, mpf, nstr, arange, linspace, sin, sqrt

mpmath.dps = 64

def outer(*args):
    if (len(args) == 0):
        yield ()
    else:
        for item in iter(args[0]):
            for items in outer(*args[1:]):
                yield (item,) + items

def make_special_vals(name, *args):
    def ctype(val):
        cls = type(val)
        if (cls is int):
            return 'int'
        elif ((cls is float) or (cls is mpf)):
            return 'double'
        elif ((cls is complex) or (cls is mpc)):
            return 'dynd_complex<double>'

    def signature(name):
        return 'dynd::nd::array {}()'.format(name)

    def static_array(name, vals, n = 15):
        return '    static {} {}[{}] = {{\n        {}\n    }};'.format(ctype(vals[0]), name, len(vals),
            ',\n        '.join(nstr(val, n) for val in vals))

    def ndarray(size, *args):
        return 'dynd::nd::array vals = dynd::nd::make_strided_array({}, dynd::ndt::make_tuple({}));'.format(size,
            ', '.join('dynd::ndt::make_type<{}>()'.format(str(arg)) for arg in args))

    ctypes = [ctype(vals[0]) for (fname, vals) in args]

    code = signature(name) + ' {\n'
    for (name, vals) in args:
        code += static_array(name, vals) + '\n'
    code += '\n'
    code += '    ' + ndarray(len(vals), *ctypes) + '\n'
    for i, (name, vals) in enumerate(args):
        code += '    vals(dynd::irange(), {}).vals() = {};\n'.format(i, name)

    code += '\n    return vals;\n}\n'

    return code

def make_factorial_vals():
    from mpmath import fac

    n = range(150)
    fac = [fac(val) for val in n]

    return make_special_vals('factorial_vals', ('n', n), ('fac', fac))

def make_factorial2_vals():
    from mpmath import fac2

    n = range(150)
    fac2 = [fac2(val) for val in n]

    return make_special_vals('factorial2_vals', ('n', n), ('fac2', fac2))

def make_factorial_ratio_vals():
    from mpmath import fac

    def fac_ratio(m, n):
        return fac(m) / fac(n)

    m = range(150)
    n = range(150)

    m, n = zip(*outer(m, n))
    fac_ratio = [fac_ratio(*vals) for vals in zip(m, n)]

    return make_special_vals('factorial_ratio_vals', ('m', m), ('n', n), ('fac_ratio', fac_ratio))

def make_gamma_vals():
    from mpmath import gamma

    x = [-mpf('0.5'), -mpf('0.01'), mpf('0.01')] + linspace(mpf('0.1'), 10, 100) + [mpf(20), mpf(30)]
    ga = [gamma(val) for val in x]

    return make_special_vals('gamma_vals', ('x', x), ('ga', ga))

def make_lgamma_vals():
    from mpmath import loggamma

    x = [mpf('0.01')] + linspace(mpf('0.1'), 10, 100) + [mpf(20), mpf(30)]
    lga = [loggamma(val) for val in x]

    return make_special_vals('lgamma_vals', ('x', x), ('lga', lga))

def make_bessel_j0_vals():
    from mpmath import besselj

    x = linspace(-5, 15, 21)
    j0 = [besselj(0, val) for val in x]

    return make_special_vals('bessel_j0_vals', ('x', x), ('j0', j0))

def make_bessel_j1_vals():
    from mpmath import besselj

    x = linspace(-5, 15, 21)
    j1 = [besselj(1, val) for val in x]

    return make_special_vals('bessel_j1_vals', ('x', x), ('j1', j1))

def make_bessel_j_vals():
    from mpmath import besselj

    nu = [mpf('0.5'), mpf('1.25'), mpf('1.5'), mpf('1.75'), mpf(2), mpf('2.75'), mpf(5), mpf(10), mpf(20)]
    x = [mpf('0.2'), mpf(1), mpf(2), mpf('2.5'), mpf(3), mpf(5), mpf(10), mpf(50)]

    nu, x = zip(*outer(nu, x))
    j = [besselj(*vals) for vals in zip(nu, x)]

    return make_special_vals('bessel_j_vals', ('nu', nu), ('x', x), ('j', j))

def make_sph_bessel_j0_vals():
    def j0(x):
        if (x == 0):
            return mpf(1)

        return sin(x) / x

    x = linspace(0, 15, 16)
    j0 = [j0(val) for val in x]

    return make_special_vals('sph_bessel_j0_vals', ('x', x), ('j0', j0))

def make_bessel_y0_vals():
    from mpmath import bessely

    x = [mpf('0.1')] + linspace(1, 15, 15)
    y0 = [bessely(0, val) for val in x]

    return make_special_vals('bessel_y0_vals', ('x', x), ('y0', y0))

def make_bessel_y1_vals():
    from mpmath import bessely

    x = [mpf('0.1')] + linspace(1, 15, 15)
    y1 = [bessely(1, val) for val in x]

    return make_special_vals('bessel_y1_vals', ('x', x), ('y1', y1))

def make_bessel_y_vals():
    from mpmath import bessely

    nu = [mpf('0.5'), mpf('1.25'), mpf('1.5'), mpf('1.75'), mpf(2), mpf('2.75'), mpf(5), mpf(10), mpf(20)]
    x = [mpf('0.2'), mpf(1), mpf(2), mpf('2.5'), mpf(3), mpf(5), mpf(10), mpf(50)]

    nu, x = zip(*outer(nu, x))
    y = [bessely(*vals) for vals in zip(nu, x)]

    return make_special_vals('bessel_y_vals', ('nu', nu), ('x', x), ('y', y))

def make_struve_h_vals():
    from mpmath import struveh

    nu = [mpf('0.5'), mpf('1.25'), mpf('1.5'), mpf('1.75'), mpf(2), mpf('2.75')]
    x = [mpf('0.2'), mpf(1), mpf(2), mpf('2.5'), mpf(3), mpf(5), mpf(10)]

    nu, x = zip(*outer(nu, x))
    h = [struveh(*vals) for vals in zip(nu, x)]

    return make_special_vals('struve_h_vals', ('nu', nu), ('x', x), ('h', h))

outfile = open('special_vals.hpp', 'w')

outfile.write('//\n')
outfile.write('// Copyright (C) 2011-14 Irwin Zaid, DyND Developers\n')
outfile.write('// BSD 2-Clause License, see LICENSE.txt\n')
outfile.write('//\n')
outfile.write('\n')
outfile.write('#ifndef _DYND__SPECIAL_VALS_HPP_\n')
outfile.write('#define _DYND__SPECIAL_VALS_HPP_\n')
outfile.write('\n')
outfile.write('#include <dynd/array.hpp>\n')
outfile.write('#include <dynd/types/tuple_type.hpp>\n')
outfile.write('\n')
outfile.write(make_factorial_vals())
outfile.write('\n')
outfile.write(make_factorial2_vals())
outfile.write('\n')
outfile.write(make_factorial_ratio_vals())
outfile.write('\n')
outfile.write(make_gamma_vals())
outfile.write('\n')
outfile.write(make_lgamma_vals())
outfile.write('\n')
outfile.write(make_bessel_j0_vals())
outfile.write('\n')
outfile.write(make_bessel_j1_vals())
outfile.write('\n')
outfile.write(make_bessel_j_vals())
outfile.write('\n')
outfile.write(make_sph_bessel_j0_vals())
outfile.write('\n')
outfile.write(make_bessel_y0_vals())
outfile.write('\n')
outfile.write(make_bessel_y1_vals())
outfile.write('\n')
outfile.write(make_bessel_y_vals())
outfile.write('\n')
outfile.write(make_struve_h_vals())
outfile.write('\n')
outfile.write('#endif // _DYND__SPECIAL_VALS_HPP_')

outfile.close()
