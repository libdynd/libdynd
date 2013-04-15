//
// Copyright (C) 2011-13, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <iostream>
#include <sstream>
#include <stdexcept>
#include "inc_gtest.hpp"

#include <dynd/ndobject.hpp>

using namespace std;
using namespace dynd;

TEST(ComplexDType, Create) {
    ndobject n;
    
    n = complex<float>(1.5f, 2.0f);
    EXPECT_EQ(n.get_dtype(), make_dtype<complex<float> >());
    EXPECT_EQ(complex<float>(1.5f, 2.0f), n.as<complex<float> >());

    n = complex<double>(2.5, 3.0);
    EXPECT_EQ(n.get_dtype(), make_dtype<complex<double> >());
    EXPECT_EQ(complex<double>(2.5, 3.0), n.as<complex<double> >());
}

TEST(ComplexDType, Properties) {
    ndobject n;
    
    n = complex<float>(1.5f, 2.0f);
    EXPECT_EQ(1.5f, n.p("real").as<float>());
    EXPECT_EQ(2.0f, n.p("imag").as<float>());

    n = complex<double>(2.5, 3.0);
    EXPECT_EQ(2.5, n.p("real").as<double>());
    EXPECT_EQ(3.0, n.p("imag").as<double>());
}
