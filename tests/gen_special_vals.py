import mpmath

from mpmath import pi, mpc, mpf, nstr, arange, linspace, sin, sqrt

mpmath.dps = 128

def outer(*iterables):
    if (len(iterables) == 0):
        yield ()
    else:
        for item in iter(iterables[0]):
            for items in outer(*iterables[1:]):
                yield (item,) + items

def make_special_vals(name, *args):
    def nstr2(obj, n):
        if hasattr(obj, '__iter__'):
            return '{' + ', '.join(nstr2(val, n) for val in obj) + '}'
        elif ((type(obj) == complex) or (type(obj) == mpc)):
            return 'dynd::dynd_complex<double>({}, {})'.format(nstr2(obj.real, n), nstr2(obj.imag, n))
        else:
            return nstr(obj, n)

    def ctype(val):
        cls = type(val)
        if (cls is int):
            return 'int'
        elif ((cls is float) or (cls is mpf)):
            return 'double'
        elif ((cls is complex) or (cls is mpc)):
            return 'dynd_complex<double>'
        else:
            return '{}[{}]'.format(ctype(val[0]), len(val))

    def dtype(val):
        cls = type(val)
        if (cls == int):
            return 'int'
        elif ((cls == float) or (cls == mpf)):
            return 'double'
        elif ((cls == complex) or (cls == mpc)):
            return 'dynd::dynd_complex<double> '
        else:
            return dtype(val[0])

        raise Exception('')

    def extents(val):
        try:
            return '[{}]'.format(len(val)) + extents(val[0])
        except TypeError:
            return ''

    def signature(name):
        return 'dynd::nd::array {}()'.format(name)

    def static_array(name, vals, n = 15):
        return '    static {} {}{} = {{\n        {}\n    }};'.format(dtype(vals), name, extents(vals),
            ',\n        '.join(nstr2(val, n) for val in vals))

    def make_type(iterable):
        if extents(iterable[0]):
            return 'dynd::ndt::cfixed_dim_from_array<{}>::make()'.format(dtype(iterable) + extents(iterable[0]))
        else:
            return 'dynd::ndt::make_type<{}>()'.format(dtype(iterable))

    def ndarray2(*iterables):
        return 'dynd::nd::array vals = dynd::nd::make_strided_array({}, dynd::ndt::make_tuple({}));'.format(len(iterables[0]),
            ', '.join(make_type(iterable) for iterable in iterables))

    def ndarray(size, *args):
        return 'dynd::nd::array vals = dynd::nd::make_strided_array({}, dynd::ndt::make_tuple({}));'.format(size,
            ', '.join('dynd::ndt::make_type<{}>()'.format(str(arg)) for arg in args))

    ctypes = [ctype(vals[0]) for (fname, vals) in args]
    iterables = [iterable for (fname, iterable) in args]

    code = signature(name) + ' {\n'
    for (name, vals) in args:
        code += static_array(name, vals) + '\n'
    code += '\n'
    code += '    ' + ndarray2(*iterables) + '\n'
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

def make_airy_vals():
    from mpmath import airyai, airybi

    x = linspace(-5, 15, 21)
    ai = [airyai(val) for val in x]
    aip = [airyai(val, 1) for val in x]
    bi = [airybi(val) for val in x]
    bip = [airybi(val, 1) for val in x]

    aibi = zip(zip(ai, aip), zip(bi, bip))

    return make_special_vals('airy_vals', ('x', x), ('aibi', aibi))

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

    nu = [mpf('0.5'), mpf('1.25'), mpf('1.5'), mpf('1.75'), mpf(2), mpf('2.75'),
        mpf(5), mpf(10), mpf(20)]
    x = [mpf('0.2'), mpf(1), mpf(2), mpf('2.5'), mpf(3), mpf(5), mpf(10), mpf(50)]

    nu, x = zip(*outer(nu, x))
    j = [besselj(*vals) for vals in zip(nu, x)]

    return make_special_vals('bessel_j_vals', ('nu', nu), ('x', x), ('j', j))

def sphbesselj(nu, x):
    from mpmath import besselj

    if (x == 0):
        if (nu == 0):
            return mpf(1)
        else:
            return mpf(0)

    return sqrt(pi / (2 * x)) * besselj(nu + mpf('0.5'), x)

def make_sph_bessel_j0_vals():
    x = linspace(0, 15, 16)
    j0 = [sphbesselj(0, val) for val in x]

    return make_special_vals('sph_bessel_j0_vals', ('x', x), ('j0', j0))

def make_sph_bessel_j_vals():
    nu = [mpf(0), mpf('0.5'), mpf('1.25'), mpf('1.5'), mpf('1.75'), mpf(2), mpf('2.75'),
        mpf(5), mpf(10), mpf(20)]
    x = [mpf(0), mpf('0.2'), mpf('0.8'), mpf(1), mpf(2), mpf('2.5'), mpf(3), mpf(5),
        mpf(10), mpf(50)]

    nu, x = zip(*outer(nu, x))
    j = [sphbesselj(*vals) for vals in zip(nu, x)]

    return make_special_vals('sph_bessel_j_vals', ('nu', nu), ('x', x), ('j', j))

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

def sphbessely(nu, x):
    from mpmath import bessely

    return sqrt(pi / (2 * x)) * bessely(nu + mpf('0.5'), x)

def make_sph_bessel_y0_vals():
    x = linspace(1, 15, 15)
    y0 = [sphbessely(0, val) for val in x]

    return make_special_vals('sph_bessel_y0_vals', ('x', x), ('y0', y0))

def make_sph_bessel_y_vals():
    nu = [mpf(0), mpf('0.5'), mpf('1.25'), mpf('1.5'), mpf('1.75'), mpf(2), mpf('2.75'),
        mpf(5), mpf(10), mpf(20)]
    x = [mpf('0.2'), mpf('0.8'), mpf(1), mpf(2), mpf('2.5'), mpf(3), mpf(5),
        mpf(10), mpf(50)]

    nu, x = zip(*outer(nu, x))
    y = [sphbessely(*vals) for vals in zip(nu, x)]

    return make_special_vals('sph_bessel_y_vals', ('nu', nu), ('x', x), ('y', y))

def make_hankel_h1_vals():
    from mpmath import hankel1

    nu = [mpf('0.5'), mpf('1.25'), mpf('1.5'), mpf('1.75'), mpf(2), mpf('2.75'),
        mpf(5), mpf(10), mpf(20)]
    x = [mpf('0.2'), mpf(1), mpf(2), mpf('2.5'), mpf(3), mpf(5), mpf(10), mpf(50)]

    nu, x = zip(*outer(nu, x))
    h1 = [hankel1(*vals) for vals in zip(nu, x)]

    return make_special_vals('hankel_h1_vals', ('nu', nu), ('x', x), ('h1', h1))

def sphhankel1(nu, x):
    return sphbesselj(nu, x) + mpc(0, 1) * sphbessely(nu, x)

def make_sph_hankel_h1_vals():
    nu = [mpf(0), mpf('0.5'), mpf('1.25'), mpf('1.5'), mpf('1.75'), mpf(2), mpf('2.75'),
        mpf(5), mpf(10), mpf(20)]
    x = [mpf('0.2'), mpf('0.8'), mpf(1), mpf(2), mpf('2.5'), mpf(3), mpf(5),
        mpf(10), mpf(50)]

    nu, x = zip(*outer(nu, x))
    h1 = [sphhankel1(*vals) for vals in zip(nu, x)]

    return make_special_vals('sph_hankel_h1_vals', ('nu', nu), ('x', x), ('h1', h1))

def make_hankel_h2_vals():
    from mpmath import hankel2

    nu = [mpf('0.5'), mpf('1.25'), mpf('1.5'), mpf('1.75'), mpf(2), mpf('2.75'),
        mpf(5), mpf(10), mpf(20)]
    x = [mpf('0.2'), mpf(1), mpf(2), mpf('2.5'), mpf(3), mpf(5), mpf(10), mpf(50)]

    nu, x = zip(*outer(nu, x))
    h2 = [hankel2(*vals) for vals in zip(nu, x)]

    return make_special_vals('hankel_h2_vals', ('nu', nu), ('x', x), ('h2', h2))

def sphhankel2(nu, x):
    return sphbesselj(nu, x) - mpc(0, 1) * sphbessely(nu, x)

def make_sph_hankel_h2_vals():
    nu = [mpf(0), mpf('0.5'), mpf('1.25'), mpf('1.5'), mpf('1.75'), mpf(2), mpf('2.75'),
        mpf(5), mpf(10), mpf(20)]
    x = [mpf('0.2'), mpf('0.8'), mpf(1), mpf(2), mpf('2.5'), mpf(3), mpf(5),
        mpf(10), mpf(50)]

    nu, x = zip(*outer(nu, x))
    h2 = [sphhankel2(*vals) for vals in zip(nu, x)]

    return make_special_vals('sph_hankel_h2_vals', ('nu', nu), ('x', x), ('h2', h2))

def make_struve_h_vals():
    from mpmath import struveh

    nu = [mpf('0.5'), mpf('1.25'), mpf('1.5'), mpf('1.75'), mpf(2), mpf('2.75')]
    x = [mpf('0.2'), mpf(1), mpf(2), mpf('2.5'), mpf(3), mpf(5), mpf(10)]

    nu, x = zip(*outer(nu, x))
    h = [struveh(*vals) for vals in zip(nu, x)]

    return make_special_vals('struve_h_vals', ('nu', nu), ('x', x), ('h', h))

def make_legendre_p_vals():
    from mpmath import legendre

    l = range(5)
    x = linspace('-0.99', '0.99', 67)

    l, x = zip(*outer(l, x))
    p = [legendre(*vals) for vals in zip(l, x)]

    return make_special_vals('legendre_p_vals', ('l', l), ('x', x), ('p', p))

def make_assoc_legendre_p_vals():
    from mpmath import legenp

    l = range(5)
    m = range(-4, 5)
    x = linspace('-0.99', '0.99', 67)

    l, m, x = zip(*(vals for vals in outer(l, m, x) if abs(vals[1]) <= vals[0]))
    p = [legenp(*vals, zeroprec = 1024) for vals in zip(l, m, x)]

    return make_special_vals('assoc_legendre_p_vals', ('l', l), ('m', m), ('x', x), ('p', p))

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
outfile.write('#include <dynd/types/cfixed_dim_type.hpp>\n')
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
outfile.write(make_airy_vals())
outfile.write('\n')
outfile.write(make_bessel_j0_vals())
outfile.write('\n')
outfile.write(make_bessel_j1_vals())
outfile.write('\n')
outfile.write(make_bessel_j_vals())
outfile.write('\n')
outfile.write(make_sph_bessel_j0_vals())
outfile.write('\n')
outfile.write(make_sph_bessel_j_vals())
outfile.write('\n')
outfile.write(make_bessel_y0_vals())
outfile.write('\n')
outfile.write(make_bessel_y1_vals())
outfile.write('\n')
outfile.write(make_bessel_y_vals())
outfile.write('\n')
outfile.write(make_sph_bessel_y0_vals())
outfile.write('\n')
outfile.write(make_sph_bessel_y_vals())
outfile.write('\n')
outfile.write(make_hankel_h1_vals())
outfile.write('\n')
outfile.write(make_sph_hankel_h1_vals())
outfile.write('\n')
outfile.write(make_hankel_h2_vals())
outfile.write('\n')
outfile.write(make_sph_hankel_h2_vals())
outfile.write('\n')
outfile.write(make_struve_h_vals())
outfile.write('\n')
outfile.write(make_legendre_p_vals())
outfile.write('\n')
outfile.write(make_assoc_legendre_p_vals())
outfile.write('\n')
outfile.write('#endif // _DYND__SPECIAL_VALS_HPP_')

outfile.close()
