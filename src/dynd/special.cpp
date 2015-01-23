//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/special.hpp>

using namespace dynd;

double dynd::factorial(int n) {
    if (n < 0) {
        throw std::invalid_argument("factorial: n must be a nonnegative integer");
    }

    double res = 1.0;
    for (int k = 1; k <= n; ++k) {
        res *= k;
    }

    return res;
}

double dynd::factorial2(int n) {
    if (n < 0) {
        throw std::invalid_argument("factorial2: n must be a nonnegative integer");
    }

    double res = 1.0;
    for (int k = n; k > 0; k -= 2) {
        res *= k;
    }

    return res;
}

double dynd::factorial_ratio(int m, int n) {
    if (m < 0 || n < 0) {
        throw std::invalid_argument("factorial_ratio: m and n must be nonnegative integers");
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

double dynd::sph_bessel_j(double nu, double x) {
    if (nu == 0) {
        return sph_bessel_j0(x);
    }

    if (x == 0.0) {
        return 0.0;
    }

    if (x < 1) {
        double x_div_2 = x / 2.0;
        double x_sq_div_4 = x_div_2 * x_div_2;

        int k = 0;
        double term = std::pow(x_div_2, nu) / gamma(nu + 1.5), res = term;
        do {
            ++k;
            term *= -x_sq_div_4 / (k * (k + nu + 0.5));
            res += term;
        } while (fabs(std::numeric_limits<double>::epsilon() * res) < fabs(term));

        return std::sqrt(dynd::_pi_by_4<double>()) * res;
    }

    return std::sqrt(dynd::_pi_by_2<double>() / x) * bessel_j(nu + 0.5, x);
}

double dynd::legendre_p_next(int l, double x, double pls1, double pl) {
    return ((2 * l + 1) * x * pl - l * pls1) / (l + 1);
}

double dynd::legendre_p(int l, double x) {
    if (l < 0) {
        throw std::invalid_argument("legendre_p: l must be a nonnegative integer");
    }

    if (fabs(x) > 1) {
        throw std::invalid_argument("legendre_p: fabs(x) must be less than or equal to 1");
    }

    double pls2, pls1 = 1.0;

    if (l == 0) {
        return pls1;
    }

    double pl = x;
    for (int k = 1; k < l; ++k) {
        pls2 = pls1;
        pls1 = pl;

        pl = legendre_p_next(k, x, pls2, pls1);
    }

    return pl;
}

double dynd::assoc_legendre_p_next(int l, int m, double x, double pl, double pls1) {
    return ((2 * l + 1) * x * pl - (l + m) * pls1) / (l - m + 1);
}

double dynd::assoc_legendre_p(int l, int m, double x) {
    if (l < 0) {
        throw std::invalid_argument("assoc_legendre_p: l must be a nonnegative integer");
    }

    if (m > l) {
        throw std::invalid_argument("assoc_legendre_p: fabs(m) must be less than or equal to l");
    }

    if (fabs(x) > 1) {
        throw std::invalid_argument("assoc_legendre_p: fabs(x) must be less than or equal to 1");
    }

    if (m == 0) {
        return legendre_p(l, x);
    }

    if (m < 0) {
        double plm = factorial_ratio(l + m, l - m) * assoc_legendre_p(l, -m, x);
        if (m % 2) {
            plm *= -1;
        }

        return plm;
    }

    double pls2m, pls1m = factorial2(2 * m - 1) * std::pow(1.0 - x * x, fabs((double) m) / 2.0);
    if (m % 2) {
        pls1m *= -1;
    }

    if (m == l) {
        return pls1m;
    }

    double p1m = x * (2 * m + 1) * pls1m;
    for (int k = m + 1; k < l; ++k) {
        pls2m = pls1m;
        pls1m = p1m;

        p1m = assoc_legendre_p_next(k, m, x, pls1m, pls2m);
    }

    return p1m;
}
