#
# Copyright (C) 2011-15 DyND Developers
# BSD 2-Clause License, see LICENSE.txt
#

import mpmath
mpmath.dps = 15

from mpmath import pi, linspace, mpc, mpf, nstr, sin, sqrt

pdps = 15

def make_special_vals(func_name, *args):
    def cstr(obj):
        try:
            return '{' + ', '.join(cstr(val) for val in obj) + '}'
        except TypeError:
            cls = type(obj)
            if ((cls == complex) or (cls == mpc)):
                return 'dynd::complex<double>({}, {})'.format(cstr(obj.real), cstr(obj.imag))

            return nstr(obj, pdps)

    def ctype(obj):
        cls = type(obj)
        if (cls == int):
            return 'int'
        elif ((cls == float) or (cls == mpf)):
            return 'double'
        elif ((cls == complex) or (cls == mpc)):
            return 'dynd::complex<double>'

        return ctype(obj[0])

    def dims(obj):
        try:
            return '[' + str(len(obj)) + ']' + dims(obj[0])
        except TypeError:
            return ''

    def make_type(vals):
        prefix, suffix = ctype(vals), dims(vals[0])
        if (suffix == ''):
            if (prefix[-1] == '>'):
                prefix += ' '

            return 'dynd::ndt::make_type<{}>()'.format(prefix)

        return 'dynd::ndt::make_type<{}>()'.format(prefix + suffix)

    def decl_asgn_static_array(name, vals):
        return 'static {} {}{} = {{\n        {}\n    }};\n'.format(ctype(vals), name, dims(vals),
            ',\n        '.join(cstr(val) for val in vals))

    def decl_asgn_ndarray(*args):
        size = len(args[0])

        return 'dynd::nd::array vals = dynd::nd::empty({}, dynd::ndt::tuple_type::make({{{}}}));\n'.format(size,
            ', '.join(make_type(vals) for vals in args))

    def asgn_vals(index, name):
        return 'vals(dynd::irange(), {}).vals() = {};\n'.format(index, name)

    names, iterables = zip(*args)

    func_def = 'dynd::nd::array {}() {{\n'.format(func_name)
    for name, vals in args:
        func_def += '    ' + decl_asgn_static_array(name, vals)
    func_def += '\n'
    func_def += '    ' + decl_asgn_ndarray(*iterables)
    for index, name in enumerate(names):
        func_def += '    ' + asgn_vals(index, name)
    func_def += '\n'
    func_def += '    return vals;\n'
    func_def += '}\n'

    return func_def

def outer(*iterables):
    if (len(iterables) == 0):
        yield ()
    else:
        for item in iter(iterables[0]):
            for items in outer(*iterables[1:]):
                yield (item,) + items

def make_factorial_vals():
    from mpmath import fac

    n = list(range(150))
    fac = [fac(val) for val in n]

    return make_special_vals('factorial_vals', ('n', n), ('fac', fac))

def make_factorial2_vals():
    from mpmath import fac2

    n = list(range(150))
    fac2 = [fac2(val) for val in n]

    return make_special_vals('factorial2_vals', ('n', n), ('fac2', fac2))

def make_factorial_ratio_vals():
    from mpmath import fac

    def fac_ratio(m, n):
        return fac(m) / fac(n)

    m = list(range(150))
    n = list(range(150))

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

    aibi = list(zip(list(zip(ai, aip)), list(zip(bi, bip))))

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

    nu, x = list(zip(*outer(nu, x)))
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

    l = list(range(5))
    x = linspace('-0.99', '0.99', 67)

    l, x = zip(*outer(l, x))
    p = [legendre(*vals) for vals in zip(l, x)]

    return make_special_vals('legendre_p_vals', ('l', l), ('x', x), ('p', p))

def make_assoc_legendre_p_vals():
    from mpmath import legenp

    l = list(range(5))
    m = list(range(-4, 5))
    x = linspace('-0.99', '0.99', 67)

    l, m, x = zip(*(vals for vals in outer(l, m, x) if abs(vals[1]) <= vals[0]))
    p = [legenp(*vals, zeroprec = 1024) for vals in zip(l, m, x)]

    return make_special_vals('assoc_legendre_p_vals', ('l', l), ('m', m), ('x', x), ('p', p))

outfile = open('special_vals.hpp', 'w')

outfile.write('//\n')
outfile.write('// Copyright (C) 2011-15 DyND Developers\n')
outfile.write('// BSD 2-Clause License, see LICENSE.txt\n')
outfile.write('//\n')
outfile.write('\n')
outfile.write('#ifndef _DYND__SPECIAL_VALS_HPP_\n')
outfile.write('#define _DYND__SPECIAL_VALS_HPP_\n')
outfile.write('\n')
outfile.write('#include <dynd/array.hpp>\n')
outfile.write('#include <dynd/types/fixed_dim_type.hpp>\n')
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
outfile.write('\n')

outfile.close()
