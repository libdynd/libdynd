//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/func/fft.hpp>

using namespace std;
using namespace dynd;

#ifdef DYND_FFTW

nd::decl::fftw nd::fftw;
nd::decl::rfftw nd::rfftw;
nd::decl::ifftw nd::ifftw;
nd::decl::irfftw nd::irfftw;

nd::decl::fft nd::fft;
nd::decl::rfft nd::rfft;
nd::decl::ifft nd::ifft;
nd::decl::irfft nd::irfft;

#endif