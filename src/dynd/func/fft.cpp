//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/func/fft.hpp>

using namespace std;
using namespace dynd;

nd::arrfunc nd::fft::make()
{
  std::vector<nd::arrfunc> children;

#ifdef DYND_FFTW
  typedef fftw_ck<fftw_complex, fftw_complex, FFTW_FORWARD> CKT;
  children.push_back(nd::as_arrfunc<CKT>());
#endif

#ifdef DYND_CUDA
  children.push_back(nd::as_arrfunc<
      cufft_ck<cufftDoubleComplex, cufftDoubleComplex, CUFFT_FORWARD>>());
#endif

  if (children.empty()) {
    throw std::runtime_error("no fft enabled");
  }

  return functional::multidispatch(
      ndt::type("(M[Fixed**N * complex[float64]], shape: ?N * int64, axes: "
                "?Fixed * int64, "
                "flags: ?int32) -> M[Fixed**N * complex[float64]]"),
      children);
}

nd::arrfunc nd::ifft::make()
{
  std::vector<nd::arrfunc> children;

#ifdef DYND_FFTW
  children.push_back(nd::as_arrfunc<
      fftw_ck<fftw_complex, fftw_complex, FFTW_BACKWARD>>());
#endif

#ifdef DYND_CUDA
  children.push_back(nd::as_arrfunc<
      cufft_ck<cufftDoubleComplex, cufftDoubleComplex, CUFFT_INVERSE>>());
#endif

  if (children.empty()) {
    throw std::runtime_error("no fft enabled");
  }

  return functional::multidispatch(
      ndt::type("(M[Fixed**N * complex[float64]], shape: ?N * int64, "
                "axes: ?Fixed * int64, "
                "flags: ?int32) -> M[Fixed**N * complex[float64]]"),
      children);
}

nd::arrfunc nd::rfft::make()
{
#ifdef DYND_FFTW
  return nd::as_arrfunc<fftw_ck<fftw_complex, double>>();
#else
  throw std::runtime_error("no fft enabled");
#endif
}

nd::arrfunc nd::irfft::make()
{
#ifdef DYND_FFTW
  return nd::as_arrfunc<fftw_ck<double, fftw_complex>>();
#else
  throw std::runtime_error("no fft enabled");
#endif
}

struct nd::fft nd::fft;
struct nd::rfft nd::rfft;

struct nd::ifft nd::ifft;
struct nd::irfft nd::irfft;