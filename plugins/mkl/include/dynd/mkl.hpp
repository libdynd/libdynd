//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <mkl.h>

#include <dynd/callable.hpp>

namespace dynd {

template <typename T>
struct mkl_type;

template <>
struct mkl_type<float> {
  typedef float type;
};

template <>
struct mkl_type<double> {
  typedef double type;
};

template <>
struct mkl_type<complex<float>> {
  typedef MKL_Complex8 type;
};

template <>
struct mkl_type<complex<double>> {
  typedef MKL_Complex16 type;
};

template <typename T>
using mkl_type_t = typename mkl_type<T>::type;

namespace nd {
  namespace mkl {

    extern callable fft;
    extern callable ifft;

    extern callable conv;

  } // namespace dynd::nd::mkl
} // namespace dynd::nd
} // namespace dynd
