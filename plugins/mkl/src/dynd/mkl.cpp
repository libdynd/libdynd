//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/callables/fft_callable.hpp>
#include <dynd/callables/ifft_callable.hpp>
#include <dynd/mkl.hpp>

using namespace std;
using namespace dynd;

nd::callable nd::mkl::fft = nd::make_callable<nd::mkl::fft_callable>();
nd::callable nd::mkl::ifft = nd::make_callable<nd::mkl::ifft_callable<dynd::complex<double>>>();
