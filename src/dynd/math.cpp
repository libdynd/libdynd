//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/math.hpp>
#include <dynd/functional.hpp>

using namespace std;
using namespace dynd;

namespace {
// CUDA and MSVC 2015 WORKAROUND: Using these functions directly in the apply
//                                template does not compile.
double mycos(double x) { return cos(x); }
double mysin(double x) { return sin(x); }
double mytan(double x) { return tan(x); }
double myexp(double x) { return exp(x); }
} // anonymous namespace

DYND_API nd::callable nd::cos = nd::functional::elwise(nd::functional::apply<double (*)(double), &mycos>());
DYND_API nd::callable nd::sin = nd::functional::elwise(nd::functional::apply<double (*)(double), &mysin>());
DYND_API nd::callable nd::tan = nd::functional::elwise(nd::functional::apply<double (*)(double), &mytan>());
DYND_API nd::callable nd::exp = nd::functional::elwise(nd::functional::apply<double (*)(double), &myexp>());
