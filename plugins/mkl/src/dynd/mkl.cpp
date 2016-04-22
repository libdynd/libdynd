//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/mkl.hpp>
#include <dynd/callable.hpp>
#include <dynd/callables/fft_callable.hpp>

using namespace std;
using namespace dynd;

nd::callable nd::mkl::fft = nd::make_callable<nd::mkl::fft_callable>();
