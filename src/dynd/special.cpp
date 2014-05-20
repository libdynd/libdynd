//
// Copyright (C) 2011-14 Irwin Zaid, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/special.hpp>

using namespace dynd;

double dynd::legendre_p_next(int l, double x, double pls1, double pl) {
    return ((2 * l + 1) * x * pl - l * pls1) / (l + 1);
}

double dynd::legendre_p(int l, double x) {
    if (l < 0) {
        throw std::runtime_error("l must be a nonnegative integer");
    }

    if (fabs(x) > 1) {
        throw std::runtime_error("fabs(x) must be less than or equal to 1");
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
        throw std::runtime_error("l must be a nonnegative integer");
    }

    if (m > l) {
        throw std::runtime_error("fabs(m) must be less than or equal to l");
    }

    if (fabs(x) > 1) {
        throw std::runtime_error("fabs(x) must be less than or equal to 1");
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

    double pls2m, pls1m = factorial2(2 * m - 1) * std::pow(1.0 - x * x, fabs(m) / 2.0);
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
