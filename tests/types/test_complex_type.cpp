//
// Copyright (C) 2011-14 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <complex>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include "inc_gtest.hpp"
#include "dynd_assertions.hpp"

#include <dynd/array.hpp>

using namespace std;
using namespace dynd;

#define EXPECT_COMPLEX_DOUBLE_EQ(a, b) EXPECT_DOUBLE_EQ(a.real(), b.real()); \
    EXPECT_DOUBLE_EQ(a.imag(), b.imag())

#define REL_ERROR_MAX 4E-15

/*
TEST(Complex, Math) {
    dynd_complex<double> z;
    typedef std::complex<double> cdbl;
    typedef std::complex<double> cdbl;

    z = dynd_complex<double>(0.0, 0.0);
    EXPECT_DOUBLE_EQ(abs(z), abs(cdbl(z)));
    EXPECT_DOUBLE_EQ(arg(z), arg(cdbl(z)));
    EXPECT_COMPLEX_DOUBLE_EQ(exp(z), exp(cdbl(z)));
    EXPECT_COMPLEX_DOUBLE_EQ(log(z), log(cdbl(z)));
    EXPECT_COMPLEX_DOUBLE_EQ(sqrt(z), sqrt(cdbl(z)));
    EXPECT_COMPLEX_DOUBLE_EQ(pow(z, 1.0), pow(cdbl(z), 1.0));
    EXPECT_COMPLEX_DOUBLE_EQ(pow(z, 2.0), pow(cdbl(z), 2.0));
    EXPECT_COMPLEX_DOUBLE_EQ(pow(z, 3.0), pow(cdbl(z), 3.0));
    EXPECT_COMPLEX_DOUBLE_EQ(pow(z, 7.4), pow(cdbl(z), 7.4));

    z = dynd_complex<double>(1.5, 2.0);
    EXPECT_DOUBLE_EQ(abs(z), abs(cdbl(z)));
    EXPECT_DOUBLE_EQ(arg(z), arg(cdbl(z)));
    EXPECT_COMPLEX_DOUBLE_EQ(exp(z), exp(cdbl(z)));
    EXPECT_COMPLEX_DOUBLE_EQ(log(z), log(cdbl(z)));
    EXPECT_COMPLEX_DOUBLE_EQ(sqrt(z), sqrt(cdbl(z)));
    EXPECT_COMPLEX_DOUBLE_EQ(pow(z, 0.0), pow(cdbl(z), 0.0));
    EXPECT_COMPLEX_DOUBLE_EQ(pow(z, 1.0), pow(cdbl(z), 1.0));
    EXPECT_EQ_RELERR(pow(z, 2.0), dynd_complex<double>(pow(cdbl(z), 2.0)),
                     REL_ERROR_MAX);
    EXPECT_EQ_RELERR(pow(z, 3.0), dynd_complex<double>(pow(cdbl(z), 3.0)),
                     REL_ERROR_MAX);
    EXPECT_COMPLEX_DOUBLE_EQ(pow(z, 7.4), pow(cdbl(z), 7.4));
    EXPECT_COMPLEX_DOUBLE_EQ(pow(z, dynd_complex<double>(0.0, 1.0)), pow(cdbl(z), complex<double>(0.0, 1.0)));
    EXPECT_COMPLEX_DOUBLE_EQ(pow(z, dynd_complex<double>(0.0, -1.0)), pow(cdbl(z), complex<double>(0.0, -1.0)));
    EXPECT_COMPLEX_DOUBLE_EQ(pow(z, dynd_complex<double>(7.4, -6.3)), pow(cdbl(z), complex<double>(7.4, -6.3)));

    z = dynd_complex<double>(1.5, 0.0);
    EXPECT_DOUBLE_EQ(abs(z), abs(cdbl(z)));
    EXPECT_DOUBLE_EQ(arg(z), arg(cdbl(z)));
    EXPECT_COMPLEX_DOUBLE_EQ(exp(z), exp(cdbl(z)));
    EXPECT_COMPLEX_DOUBLE_EQ(log(z), log(cdbl(z)));
    EXPECT_COMPLEX_DOUBLE_EQ(sqrt(z), sqrt(cdbl(z)));
    EXPECT_COMPLEX_DOUBLE_EQ(pow(z, 0.0), pow(cdbl(z), 0.0));
    EXPECT_COMPLEX_DOUBLE_EQ(pow(z, 1.0), pow(cdbl(z), 1.0));
    EXPECT_COMPLEX_DOUBLE_EQ(pow(z, 2.0), pow(cdbl(z), 2.0));
    EXPECT_COMPLEX_DOUBLE_EQ(pow(z, 3.0), pow(cdbl(z), 3.0));
    EXPECT_COMPLEX_DOUBLE_EQ(pow(z, 7.4), pow(cdbl(z), 7.4));
    EXPECT_COMPLEX_DOUBLE_EQ(pow(z, dynd_complex<double>(0.0, 1.0)), pow(cdbl(z), complex<double>(0.0, 1.0)));
    EXPECT_COMPLEX_DOUBLE_EQ(pow(z, dynd_complex<double>(0.0, -1.0)), pow(cdbl(z), complex<double>(0.0, -1.0)));
    EXPECT_COMPLEX_DOUBLE_EQ(pow(z, dynd_complex<double>(7.4, -6.3)), pow(cdbl(z), complex<double>(7.4, -6.3)));

    z = dynd_complex<double>(0.0, 2.0);
    EXPECT_DOUBLE_EQ(abs(z), abs(cdbl(z)));
    EXPECT_DOUBLE_EQ(arg(z), arg(cdbl(z)));
    EXPECT_COMPLEX_DOUBLE_EQ(exp(z), exp(cdbl(z)));
    EXPECT_COMPLEX_DOUBLE_EQ(log(z), log(cdbl(z)));
    EXPECT_COMPLEX_DOUBLE_EQ(sqrt(z), sqrt(cdbl(z)));
    EXPECT_COMPLEX_DOUBLE_EQ(pow(z, 0.0), pow(cdbl(z), 0.0));
    EXPECT_COMPLEX_DOUBLE_EQ(pow(z, 1.0), pow(cdbl(z), 1.0));
    EXPECT_COMPLEX_DOUBLE_EQ(pow(z, 2.0), pow(cdbl(z), 2.0));
    EXPECT_COMPLEX_DOUBLE_EQ(pow(z, 3.0), pow(cdbl(z), 3.0));
    EXPECT_EQ_RELERR(pow(z, 7.4), dynd_complex<double>(pow(cdbl(z), 7.4)),
                     REL_ERROR_MAX);
    EXPECT_COMPLEX_DOUBLE_EQ(pow(z, dynd_complex<double>(0.0, 1.0)), pow(cdbl(z), complex<double>(0.0, 1.0)));
    EXPECT_COMPLEX_DOUBLE_EQ(pow(z, dynd_complex<double>(0.0, -1.0)), pow(cdbl(z), complex<double>(0.0, -1.0)));
    EXPECT_EQ_RELERR(
        pow(z, dynd_complex<double>(7.4, -6.3)),
        dynd_complex<double>(pow(cdbl(z), complex<double>(7.4, -6.3))),
        REL_ERROR_MAX);

    // Todo: pow works for both arguments complex, but there is a very small difference in the answers from dynd and std.
    // That's fine, but we need to specify a floating-point tolerance for testing.

    z = dynd_complex<double>(10.0, 0.5);
    EXPECT_DOUBLE_EQ(abs(z), abs(cdbl(z)));
    EXPECT_DOUBLE_EQ(arg(z), arg(cdbl(z)));
    EXPECT_COMPLEX_DOUBLE_EQ(exp(z), exp(cdbl(z)));
    EXPECT_COMPLEX_DOUBLE_EQ(log(z), log(cdbl(z)));
    EXPECT_COMPLEX_DOUBLE_EQ(sqrt(z), sqrt(cdbl(z)));
    EXPECT_COMPLEX_DOUBLE_EQ(cos(z), cos(cdbl(z)));
    EXPECT_COMPLEX_DOUBLE_EQ(sin(z), sin(cdbl(z)));

    z = dynd_complex<double>(10.0, -0.5);
    EXPECT_DOUBLE_EQ(abs(z), abs(cdbl(z)));
    EXPECT_DOUBLE_EQ(arg(z), arg(cdbl(z)));
    EXPECT_COMPLEX_DOUBLE_EQ(exp(z), exp(cdbl(z)));
    EXPECT_COMPLEX_DOUBLE_EQ(log(z), log(cdbl(z)));
    EXPECT_COMPLEX_DOUBLE_EQ(sqrt(z), sqrt(cdbl(z)));
    EXPECT_COMPLEX_DOUBLE_EQ(cos(z), cos(cdbl(z)));
    EXPECT_COMPLEX_DOUBLE_EQ(sin(z), sin(cdbl(z)));

    z = dynd_complex<double>(-10.0, 0.5);
    EXPECT_DOUBLE_EQ(abs(z), abs(cdbl(z)));
    EXPECT_DOUBLE_EQ(arg(z), arg(cdbl(z)));
    EXPECT_COMPLEX_DOUBLE_EQ(exp(z), exp(cdbl(z)));
    EXPECT_COMPLEX_DOUBLE_EQ(log(z), log(cdbl(z)));
    EXPECT_COMPLEX_DOUBLE_EQ(sqrt(z), sqrt(cdbl(z)));
    EXPECT_COMPLEX_DOUBLE_EQ(cos(z), cos(cdbl(z)));
    EXPECT_COMPLEX_DOUBLE_EQ(sin(z), sin(cdbl(z)));

    z = dynd_complex<double>(-10.0, -0.5);
    EXPECT_DOUBLE_EQ(abs(z), abs(cdbl(z)));
    EXPECT_DOUBLE_EQ(arg(z), arg(cdbl(z)));
    EXPECT_COMPLEX_DOUBLE_EQ(exp(z), exp(cdbl(z)));
    EXPECT_COMPLEX_DOUBLE_EQ(log(z), log(cdbl(z)));
    EXPECT_COMPLEX_DOUBLE_EQ(sqrt(z), sqrt(cdbl(z)));
    EXPECT_COMPLEX_DOUBLE_EQ(cos(z), cos(cdbl(z)));
    EXPECT_COMPLEX_DOUBLE_EQ(sin(z), sin(cdbl(z)));
}
*/

#undef ASSERT_COMPLEX_DOUBLE_EQ

TEST(ComplexDType, Create) {
    ndt::type d;

    // complex[float32]
    d = ndt::make_type<dynd_complex<float> >();
    EXPECT_EQ(complex_float32_type_id, d.get_type_id());
    EXPECT_EQ(complex_kind, d.get_kind());
    EXPECT_EQ(8u, d.get_data_size());
    EXPECT_EQ((size_t)scalar_align_of<float>::value, d.get_data_alignment());
    EXPECT_FALSE(d.is_expression());
    EXPECT_EQ("complex[float32]", d.str());
    // Roundtripping through a string
    EXPECT_EQ(d, ndt::type(d.str()));

    // complex[float64]
    d = ndt::make_type<dynd_complex<double> >();
    EXPECT_EQ(complex_float64_type_id, d.get_type_id());
    EXPECT_EQ(complex_kind, d.get_kind());
    EXPECT_EQ(16u, d.get_data_size());
    EXPECT_EQ((size_t)scalar_align_of<double>::value, d.get_data_alignment());
    EXPECT_FALSE(d.is_expression());
    EXPECT_EQ("complex[float64]", d.str());
    // Roundtripping through a string
    EXPECT_EQ(d, ndt::type(d.str()));
}

TEST(ComplexDType, CreateFromValue) {
    nd::array n;
    
    n = dynd_complex<float>(1.5f, 2.0f);
    EXPECT_EQ(n.get_type(), ndt::make_type<dynd_complex<float> >());
    EXPECT_EQ(dynd_complex<float>(1.5f, 2.0f), n.as<dynd_complex<float> >());

    n = dynd_complex<double>(2.5, 3.0);
    EXPECT_EQ(n.get_type(), ndt::make_type<dynd_complex<double> >());
    EXPECT_EQ(dynd_complex<double>(2.5, 3.0), n.as<dynd_complex<double> >());
}

TEST(ComplexDType, Properties) {
    nd::array n;
    
    n = dynd_complex<float>(1.5f, 2.0f);
    EXPECT_EQ(1.5f, n.p("real").as<float>());
    EXPECT_EQ(2.0f, n.p("imag").as<float>());

    n = dynd_complex<double>(2.5, 3.0);
    EXPECT_EQ(2.5, n.p("real").as<double>());
    EXPECT_EQ(3.0, n.p("imag").as<double>());

    complex<double> avals[3] = {complex<double>(1, 2), complex<double>(-1, 1.5),
                                complex<double>(3, 21.75)};
    n = avals;
    EXPECT_EQ(1., n.p("real")(0).as<double>());
    EXPECT_EQ(2., n.p("imag")(0).as<double>());
    EXPECT_EQ(-1., n.p("real")(1).as<double>());
    EXPECT_EQ(1.5, n.p("imag")(1).as<double>());
    EXPECT_EQ(3., n.p("real")(2).as<double>());
    EXPECT_EQ(21.75, n.p("imag")(2).as<double>());
}
