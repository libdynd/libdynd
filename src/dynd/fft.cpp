//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/fft.hpp>
#include <dynd/func/take.hpp>
#include <dynd/functional.hpp>
#include <dynd/callables/fft_callable.hpp>

using namespace std;
using namespace dynd;

#ifdef DYND_FFTW

namespace {

nd::callable make_fft()
{
  std::vector<nd::callable> children;

  typedef nd::fftw_callable<fftw_complex, fftw_complex, FFTW_FORWARD> CKT;
  children.push_back(nd::make_callable<CKT>());

  return children[0];
  /*
    return functional::multidispatch(
        ndt::type("(M[Fixed**N * complex[float64]], shape: ?N * int64, axes: "
                  "?Fixed * int64, "
                  "flags: ?int32) -> M[Fixed**N * complex[float64]]"),
        children, {"N"});
  */
}

nd::callable make_ifft()
{
  std::vector<nd::callable> children;

  children.push_back(nd::make_callable<nd::fftw_callable<fftw_complex, fftw_complex, FFTW_BACKWARD>>());

  return children[0];
  /*
    return functional::multidispatch(
        ndt::type("(M[Fixed**N * complex[float64]], shape: ?N * int64, "
                  "axes: ?Fixed * int64, "
                  "flags: ?int32) -> M[Fixed**N * complex[float64]]"),
        children, {"N"});
  */
}

nd::callable make_rfft()
{
  return nd::make_callable<nd::fftw_callable<fftw_complex, double>>();
}

nd::callable make_irfft()
{
  return nd::make_callable<nd::fftw_callable<double, fftw_complex>>();
}

} // unnamed namespace

DYND_API nd::callable nd::fft = make_fft();
DYND_API nd::callable nd::ifft = make_ifft();
DYND_API nd::callable nd::rfft = make_rfft();
DYND_API nd::callable nd::irfft = make_irfft();

#endif

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
