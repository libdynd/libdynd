//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/callables/base_dispatch_callable.hpp>
#include <dynd/callables/conv_callable.hpp>
#include <dynd/callables/fft_callable.hpp>
#include <dynd/callables/ifft_callable.hpp>
#include <dynd/mkl.hpp>

using namespace std;
using namespace dynd;

namespace {

nd::callable make_conv() {
  class dispatch_callable : public nd::base_dispatch_callable {
    nd::callable float32_child;
    nd::callable float64_child;
    nd::callable complex64_child;
    nd::callable complex128_child;

  public:
    dispatch_callable()
        : base_dispatch_callable(ndt::type("(Fixed**N * Scalar, Fixed**N * Scalar) -> Fixed**N * Scalar")),
          float32_child(nd::make_callable<nd::mkl::conv_callable<float>>()),
          float64_child(nd::make_callable<nd::mkl::conv_callable<double>>()),
          complex64_child(nd::make_callable<nd::mkl::conv_callable<dynd::complex<float>>>()),
          complex128_child(nd::make_callable<nd::mkl::conv_callable<dynd::complex<double>>>()) {}

    const nd::callable &specialize(const ndt::type &DYND_UNUSED(dst_tp), intptr_t DYND_UNUSED(nsrc),
                                   const ndt::type *src_tp) {
      type_id_t data_tp_id = src_tp[0].get_dtype().get_id();

      switch (data_tp_id) {
      case float32_id:
        return float32_child;
      case float64_id:
        return float64_child;
      case complex_float64_id:
        return complex128_child;
      default:
        throw std::runtime_error("error");
      }
    }
  };

  return nd::make_callable<dispatch_callable>();
}

/*
int init() {
  nd::reg("conv", nd::mkl::conv);

  return 0;
}
*/

} // unnamed namespace

nd::callable nd::mkl::fft = nd::make_callable<nd::mkl::fft_callable>();
nd::callable nd::mkl::ifft = nd::make_callable<nd::mkl::ifft_callable<dynd::complex<double>>>();

nd::callable nd::mkl::conv = make_conv();
