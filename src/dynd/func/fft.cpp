//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/func/fft.hpp>
#include <dynd/func/take.hpp>
#include <dynd/functional.hpp>

using namespace std;
using namespace dynd;

DYND_API nd::callable nd::fft::make()
{
  std::vector<nd::callable> children;

#ifdef DYND_FFTW
  typedef fftw_ck<fftw_complex, fftw_complex, FFTW_FORWARD> CKT;
  children.push_back(nd::callable::make<CKT>(0));
#endif

#ifdef DYND_CUDA
  children.push_back(nd::callable::make<cufft_ck<cufftDoubleComplex, cufftDoubleComplex, CUFFT_FORWARD>>(0));
#endif

  if (children.empty()) {
    throw std::runtime_error("no fft enabled");
  }

  return children[0];
  /*
    return functional::multidispatch(
        ndt::type("(M[Fixed**N * complex[float64]], shape: ?N * int64, axes: "
                  "?Fixed * int64, "
                  "flags: ?int32) -> M[Fixed**N * complex[float64]]"),
        children, {"N"});
  */
}

DYND_API nd::callable nd::ifft::make()
{
  std::vector<nd::callable> children;

#ifdef DYND_FFTW
  children.push_back(nd::callable::make<fftw_ck<fftw_complex, fftw_complex, FFTW_BACKWARD>>());
#endif

#ifdef DYND_CUDA
  children.push_back(nd::callable::make<cufft_ck<cufftDoubleComplex, cufftDoubleComplex, CUFFT_INVERSE>>());
#endif

  if (children.empty()) {
    throw std::runtime_error("no fft enabled");
  }

  return children[0];
  /*
    return functional::multidispatch(
        ndt::type("(M[Fixed**N * complex[float64]], shape: ?N * int64, "
                  "axes: ?Fixed * int64, "
                  "flags: ?int32) -> M[Fixed**N * complex[float64]]"),
        children, {"N"});
  */
}

DYND_API nd::callable nd::rfft::make()
{
#ifdef DYND_FFTW
  return nd::callable::make<fftw_ck<fftw_complex, double>>(0);
#else
  throw std::runtime_error("no fft enabled");
#endif
}

DYND_API nd::callable nd::irfft::make()
{
#ifdef DYND_FFTW
  return nd::callable::make<fftw_ck<double, fftw_complex>>(0);
#else
  throw std::runtime_error("no fft enabled");
#endif
}

DYND_DEFAULT_DECLFUNC_GET(nd::fft)
DYND_DEFAULT_DECLFUNC_GET(nd::rfft)
DYND_DEFAULT_DECLFUNC_GET(nd::ifft)
DYND_DEFAULT_DECLFUNC_GET(nd::irfft)

DYND_API struct nd::fft nd::fft;
DYND_API struct nd::rfft nd::rfft;
DYND_API struct nd::ifft nd::ifft;
DYND_API struct nd::irfft nd::irfft;

nd::array nd::fftshift(const nd::array &x)
{
  nd::array y = x;
  for (intptr_t i = 0; i < x.get_ndim(); ++i) {
    intptr_t p = y.get_dim_size();
    intptr_t q = (p + 1) / 2;
    y = take(y, nd::concatenate(nd::range(q, p), nd::range(q)));
    y = y.rotate();
  }
  return y;
}

nd::array nd::ifftshift(const nd::array &x)
{
  nd::array y = x;
  for (intptr_t i = 0; i < x.get_ndim(); ++i) {
    intptr_t p = y.get_dim_size();
    intptr_t q = p - (p + 1) / 2;
    y = take(y, nd::concatenate(nd::range(q, p), nd::range(q)));
    y = y.rotate();
  }
  return y;
}

nd::array nd::fftspace(intptr_t count, double step)
{
  // Todo: When casting is fixed, change the ranges below to integer versions
  return nd::concatenate(nd::range((count - 1) / 2 + 1.0), nd::range(-count / 2 + 0.0, 0.0)) / (count * step);
}
