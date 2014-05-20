//
// Copyright (C) 2011-14 Irwin Zaid, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DYND__SPECIAL_HPP_
#define _DYND__SPECIAL_HPP_

#include <dynd/array.hpp>
#include <dynd/cephes.hpp>
#include <dynd/dynd_math.hpp>

namespace dynd {

inline double factorial(int n) {
    if (n < 0) {
        throw std::runtime_error("n must be a nonnegative integer");
    }

    double res = 1.0;
    for (int k = 1; k <= n; ++k) {
        res *= k;
    }

    return res;
}

inline double factorial2(int n) {
    if (n < 0) {
        throw std::runtime_error("n must be a nonnegative integer");
    }

    double res = 1.0;
    for (int k = n; k > 0; k -= 2) {
        res *= k;
    }

    return res;
}

inline double factorial_ratio(int m, int n) {
    if (m < 0 || n < 0) {
        throw std::runtime_error("m and n must be nonnegative integers");
    }

    if (m < n) {
        return 1.0 / factorial_ratio(n, m);
    }

    double res = 1.0;
    for (int k = n + 1; k <= m; ++k) {
        res *= k;
    }

    return res;
}

inline double gamma(double x) {
    return cephes::Gamma(x);
}

inline double lgamma(double x) {
    return cephes::lgam(x);
}

inline double bessel_j0(double x) {
    return cephes::j0(x);
}

inline double bessel_j1(double x) {
    return cephes::j1(x);
}

inline double bessel_j(double nu, double x) {
    return cephes::jv(nu, x);
}

inline double sph_bessel_j0(double x) {
    if (x == 0.0) {
        return 1.0;
    }

    return std::sin(x) / x;
}

inline double bessel_y0(double x) {
    return cephes::y0(x);
}

inline double bessel_y1(double x) {
    return cephes::y1(x);
}

inline double bessel_y(double nu, double x) {
    return cephes::yv(nu, x);
}

inline double struve_h(double nu, double x) {
    return cephes::struve(nu, x);
}








/*
inline void airy(double x, double &ai, double &aip, double &bi, double &bip) {
    cephes::airy(x, &ai, &aip, &bi, &bip);
}
*/

/*
inline double sph_bessel_j(int n, double x) {
    if (n == 0) {
        return sph_bessel_j0(x);
    }

    if (x == 0.0) {
        return 0.0;
    }

    if (x < 1) {
        if (

mult = x / 2;
51	if(v + 3 > max_factorial<T>::value)
52	{
53	term = v * log(mult) - boost::math::lgamma(v+1+T(0.5f), Policy());
54	term = exp(term);
55	}
56	else
57	term = pow(mult, T(v)) / boost::math::tgamma(v+1+T(0.5f), Policy());
58	mult *= -mult;


    }

    return std::sqrt(dynd_pi_div_2<double>() / x) * bessel_j(n + 0.5, x);
}
*/

inline double riccati_bessel_j0(double x) {
    return std::sin(x);
}

inline double riccati_bessel_j1(double x) {
    if (x == 0.0) {
        return 1.0 - std::cos(x);
    }

    return std::sin(x) / x - std::cos(x);
}

} // namespace dynd

#endif // _DYND__SPECIAL_HPP_
