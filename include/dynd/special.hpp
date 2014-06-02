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

double factorial(int n);
double factorial2(int n);

double factorial_ratio(int m, int n);

inline double gamma(double x) {
    return cephes_Gamma(x);
}

inline double lgamma(double x) {
    return cephes_lgam(x);
}

inline void airy(double (&res)[2][2], double x) {
    cephes_airy(x, &res[0][0], &res[0][1], &res[1][0], &res[1][1]);
}

inline void airy_ai(double (&res)[2], double x) {
    double tmp[2];
    cephes_airy(x, &res[0], &res[1], &tmp[0], &tmp[1]);
}

inline void airy_bi(double (&res)[2], double x) {
    double tmp[2];
    cephes_airy(x, &tmp[0], &tmp[1], &res[0], &res[1]);
}

inline double bessel_j0(double x) {
    return cephes_j0(x);
}

inline double bessel_j1(double x) {
    return cephes_j1(x);
}

inline double bessel_j(double nu, double x) {
    return cephes_jv(nu, x);
}

inline double sph_bessel_j0(double x) {
    if (x == 0.0) {
        return 1.0;
    }

    return std::sin(x) / x;
}

double sph_bessel_j(double nu, double x);

inline double riccati_bessel_j(double nu, double x) {
    return std::sqrt(dynd::dynd_pi_div_2<double>() * x) * bessel_j(nu + 0.5, x);
}

inline double bessel_y0(double x) {
    return cephes_y0(x);
}

inline double bessel_y1(double x) {
    return cephes_y1(x);
}

inline double bessel_y(double nu, double x) {
    return cephes_yv(nu, x);
}

inline double sph_bessel_y0(double x) {
    return -std::cos(x) / x;
}

inline double sph_bessel_y(double nu, double x) {
    return std::sqrt(dynd::dynd_pi_div_2<double>() / x) * bessel_y(nu + 0.5, x);
}

inline double riccati_bessel_y(double nu, double x) {
    return -std::sqrt(dynd::dynd_pi_div_2<double>() * x) * bessel_y(nu + 0.5, x);
}

inline dynd_complex<double> hankel_h1(double nu, double x) {
    return dynd_complex<double>(bessel_j(nu, x), bessel_y(nu, x));
}

inline dynd_complex<double> sph_hankel_h1(double nu, double x) {
    return dynd_complex<double>(sph_bessel_j(nu, x), sph_bessel_y(nu, x));
}

inline dynd_complex<double> riccati_hankel_h1(double nu, double x) {
    return dynd_complex<double>(riccati_bessel_j(nu, x), -riccati_bessel_y(nu, x));
}

inline dynd_complex<double> hankel_h2(double nu, double x) {
    return dynd_complex<double>(bessel_j(nu, x), -bessel_y(nu, x));
}

inline dynd_complex<double> sph_hankel_h2(double nu, double x) {
    return dynd_complex<double>(sph_bessel_j(nu, x), -sph_bessel_y(nu, x));
}

inline dynd_complex<double> riccati_hankel_h2(double nu, double x) {
    return dynd_complex<double>(riccati_bessel_j(nu, x), riccati_bessel_y(nu, x));
}

inline double struve_h(double nu, double x) {
    return cephes_struve(nu, x);
}

double legendre_p_next(int l, double x, double pls1, double pl);
double legendre_p(int l, double x);

double assoc_legendre_p_next(int l, int m, double x, double pl, double pls1);
double assoc_legendre_p(int l, int m, double x);

} // namespace dynd

#endif // _DYND__SPECIAL_HPP_
