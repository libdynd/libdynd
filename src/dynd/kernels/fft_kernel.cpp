//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/callable.hpp>
#include <dynd/kernels/fft_kernel.hpp>

using namespace std;
using namespace dynd;

#ifdef DYND_FFTW

::fftwf_plan nd::detail::fftw_plan_guru_dft(int rank, const fftw_iodim *dims,
                                        int howmany_rank,
                                        const fftw_iodim *howmany_dims,
                                        fftwf_complex *in, fftwf_complex *out,
                                        int sign, unsigned flags)
{
  return ::fftwf_plan_guru_dft(rank, dims, howmany_rank, howmany_dims, in, out,
                               sign, flags);
}

::fftw_plan nd::detail::fftw_plan_guru_dft(int rank, const fftw_iodim *dims,
                                       int howmany_rank,
                                       const fftw_iodim *howmany_dims,
                                       fftw_complex *in, fftw_complex *out,
                                       int sign, unsigned flags)
{
  return ::fftw_plan_guru_dft(rank, dims, howmany_rank, howmany_dims, in, out,
                              sign, flags);
}

#endif
