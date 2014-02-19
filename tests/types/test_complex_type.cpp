//
// Copyright (C) 2011-14 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <iostream>
#include <sstream>
#include <stdexcept>
#include "inc_gtest.hpp"

#include <dynd/array.hpp>

using namespace std;
using namespace dynd;

TEST(ComplexDType, Create) {
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
}
