//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/func/fft.hpp>
#include <dynd/func/multidispatch.hpp>
#include <dynd/func/take.hpp>

using namespace std;
using namespace dynd;

nd::arrfunc nd::fft::make()
{
  std::vector<nd::arrfunc> children;

#ifdef DYND_FFTW
  typedef fftw_ck<fftw_complex, fftw_complex, FFTW_FORWARD> CKT;
  children.push_back(nd::arrfunc::make<CKT>(0));
#endif

#ifdef DYND_CUDA
  children.push_back(nd::arrfunc::make<
      cufft_ck<cufftDoubleComplex, cufftDoubleComplex, CUFFT_FORWARD>>(0));
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

nd::arrfunc nd::ifft::make()
{
  std::vector<nd::arrfunc> children;

#ifdef DYND_FFTW
  children.push_back(
      nd::arrfunc::make<fftw_ck<fftw_complex, fftw_complex, FFTW_BACKWARD>>(0));
#endif

#ifdef DYND_CUDA
  children.push_back(nd::arrfunc::make<
      cufft_ck<cufftDoubleComplex, cufftDoubleComplex, CUFFT_INVERSE>>(0));
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

nd::arrfunc nd::rfft::make()
{
#ifdef DYND_FFTW
  return nd::arrfunc::make<fftw_ck<fftw_complex, double>>(0);
#else
  throw std::runtime_error("no fft enabled");
#endif
}

nd::arrfunc nd::irfft::make()
{
#ifdef DYND_FFTW
  return nd::arrfunc::make<fftw_ck<double, fftw_complex>>(0);
#else
  throw std::runtime_error("no fft enabled");
#endif
}

struct nd::fft nd::fft;
struct nd::rfft nd::rfft;

struct nd::ifft nd::ifft;
struct nd::irfft nd::irfft;

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
  return nd::concatenate(nd::range((count - 1) / 2 + 1.0),
                         nd::range(-count / 2 + 0.0, 0.0)) /
         (count * step);
}
