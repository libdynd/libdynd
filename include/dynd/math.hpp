//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/config.hpp>

namespace dynd {

template <typename T>
DYND_CUDA_HOST_DEVICE T _e(); // e

template <typename T>
DYND_CUDA_HOST_DEVICE T _log2_e(); // log2(e)

template <typename T>
DYND_CUDA_HOST_DEVICE T _log10_e(); // log10(e)

template <typename T>
DYND_CUDA_HOST_DEVICE T _log_2(); // log(2)

template <typename T>
DYND_CUDA_HOST_DEVICE T _log_10(); // log(10)

template <typename T>
DYND_CUDA_HOST_DEVICE T _pi(); // pi

template <typename T>
DYND_CUDA_HOST_DEVICE T _2_pi(); // 2 * pi

template <typename T>
DYND_CUDA_HOST_DEVICE T _pi_by_2(); // pi / 2

template <typename T>
DYND_CUDA_HOST_DEVICE T _pi_by_4(); // pi / 4

template <typename T>
DYND_CUDA_HOST_DEVICE T _1_by_pi(); // 1 / pi

template <typename T>
DYND_CUDA_HOST_DEVICE T _2_by_pi(); // 2 / pi

template <typename T>
DYND_CUDA_HOST_DEVICE T _sqrt_2(); // sqrt(2)

template <typename T>
DYND_CUDA_HOST_DEVICE T _1_by_sqrt_2(); // 1 / sqrt(2)

template <typename T>
DYND_CUDA_HOST_DEVICE T _nan(const char *arg); // nan

template <>
DYND_CUDA_HOST_DEVICE inline float _e<float>()
{
  return 2.718281828459045235360287471352662498f;
}

template <>
DYND_CUDA_HOST_DEVICE inline float _log2_e<float>()
{
  return 1.442695040888963407359924681001892137f;
}

template <>
DYND_CUDA_HOST_DEVICE inline float _log10_e<float>()
{
  return 0.434294481903251827651128918916605082f;
}

template <>
DYND_CUDA_HOST_DEVICE inline float _log_2<float>()
{
  return 0.693147180559945309417232121458176568f;
}

template <>
DYND_CUDA_HOST_DEVICE inline float _log_10<float>()
{
  return 2.302585092994045684017991454684364208f;
}

template <>
DYND_CUDA_HOST_DEVICE inline float _pi<float>()
{
  return 3.141592653589793238462643383279502884f;
}

template <>
DYND_CUDA_HOST_DEVICE inline float _2_pi<float>()
{
  return 6.283185307179586231995926937088370323f;
}

template <>
DYND_CUDA_HOST_DEVICE inline float _pi_by_2<float>()
{
  return 1.570796326794896619231321691639751442f;
}

template <>
DYND_CUDA_HOST_DEVICE inline float _pi_by_4<float>()
{
  return 0.785398163397448309615660845819875721f;
}

template <>
DYND_CUDA_HOST_DEVICE inline float _1_by_pi<float>()
{
  return 0.318309886183790671537767526745028724f;
}

template <>
DYND_CUDA_HOST_DEVICE inline float _2_by_pi<float>()
{
  return 0.636619772367581343075535053490057448f;
}

template <>
DYND_CUDA_HOST_DEVICE inline float _sqrt_2<float>()
{
  return 1.414213562373095048801688724209698079f;
}

template <>
DYND_CUDA_HOST_DEVICE inline float _1_by_sqrt_2<float>()
{
  return 0.707106781186547524400844362104849039f;
}

template <>
DYND_CUDA_HOST_DEVICE inline float _nan(const char *arg)
{
#ifdef __CUDACC__
  return ::nanf(arg);
#else
  return std::nanf(arg);
#endif
}

template <>
DYND_CUDA_HOST_DEVICE inline double _e<double>()
{
  return 2.718281828459045235360287471352662498;
}

template <>
DYND_CUDA_HOST_DEVICE inline double _log2_e<double>()
{
  return 1.442695040888963407359924681001892137;
}

template <>
DYND_CUDA_HOST_DEVICE inline double _log10_e<double>()
{
  return 0.434294481903251827651128918916605082;
}

template <>
DYND_CUDA_HOST_DEVICE inline double _log_2<double>()
{
  return 0.693147180559945309417232121458176568;
}

template <>
DYND_CUDA_HOST_DEVICE inline double _log_10<double>()
{
  return 2.302585092994045684017991454684364208;
}

template <>
DYND_CUDA_HOST_DEVICE inline double _pi<double>()
{
  return 3.141592653589793238462643383279502884;
}

template <>
DYND_CUDA_HOST_DEVICE inline double _2_pi<double>()
{
  return 6.283185307179586231995926937088370323;
}

template <>
DYND_CUDA_HOST_DEVICE inline double _pi_by_2<double>()
{
  return 1.570796326794896619231321691639751442;
}

template <>
DYND_CUDA_HOST_DEVICE inline double _pi_by_4<double>()
{
  return 0.785398163397448309615660845819875721;
}

template <>
DYND_CUDA_HOST_DEVICE inline double _1_by_pi<double>()
{
  return 0.318309886183790671537767526745028724;
}

template <>
DYND_CUDA_HOST_DEVICE inline double _2_by_pi<double>()
{
  return 0.636619772367581343075535053490057448;
}

template <>
DYND_CUDA_HOST_DEVICE inline double _sqrt_2<double>()
{
  return 1.414213562373095048801688724209698079;
}

template <>
DYND_CUDA_HOST_DEVICE inline double _1_by_sqrt_2<double>()
{
  return 0.707106781186547524400844362104849039;
}

template <>
DYND_CUDA_HOST_DEVICE inline double _nan(const char *arg)
{
#ifdef __CUDACC__
  return ::nan(arg);
#else
  return std::nan(arg);
#endif
}

#ifdef __CUDACC__
#define NAMESPACE
#else
#define NAMESPACE std
#endif

using NAMESPACE::cos;
using NAMESPACE::sin;
using NAMESPACE::tan;
using NAMESPACE::atan2;
using NAMESPACE::cosh;
using NAMESPACE::sinh;
using NAMESPACE::exp;
using NAMESPACE::log;
using NAMESPACE::pow;
using NAMESPACE::sqrt;
using NAMESPACE::cbrt;
using NAMESPACE::hypot;
using NAMESPACE::abs;
using NAMESPACE::isfinite;
using NAMESPACE::isinf;
using NAMESPACE::isnan;

#undef NAMESPACE

} // namespace dynd