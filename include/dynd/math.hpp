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

template <typename T>
DYND_CUDA_HOST_DEVICE T abs(complex<T> z)
{
  return static_cast<T>(hypot(z.real(), z.imag()));
}

template <typename T>
DYND_CUDA_HOST_DEVICE T arg(complex<T> z)
{
  return atan2(z.imag(), z.real());
}

template <typename T>
DYND_CUDA_HOST_DEVICE complex<T> exp(complex<T> z)
{
  T x, c, s;
  T r = z.real(), i = z.imag();
  complex<T> ret;

  if (isfinite(r)) {
    x = exp(r);

    c = cos(i);
    s = sin(i);

    if (isfinite(i)) {
      ret = complex<T>(x * c, x * s);
    } else {
      ret = complex<T>(_nan<T>(NULL), copysign(_nan<T>(NULL), i));
    }
  } else if (isnan(r)) {
    // r is nan
    if (i == 0) {
      ret = complex<T>(r, 0);
    } else {
      ret = complex<T>(r, copysign(_nan<T>(NULL), i));
    }
  } else {
    // r is +- inf
    if (r > 0) {
      if (i == 0) {
        ret = complex<T>(r, i);
      } else if (isfinite(i)) {
        c = cos(i);
        s = sin(i);

        ret = complex<T>(r * c, r * s);
      } else {
        // x = +inf, y = +-inf | nan
        ret = complex<T>(r, _nan<T>(NULL));
      }
    } else {
      if (isfinite(i)) {
        x = exp(r);
        c = cos(i);
        s = sin(i);

        ret = complex<T>(x * c, x * s);
      } else {
        // x = -inf, y = nan | +i inf
        ret = complex<T>(0, 0);
      }
    }
  }

  return ret;
}

template <typename T>
DYND_CUDA_HOST_DEVICE complex<T> log(complex<T> z)
{
  return complex<T>(log(abs(z)), arg(z));
}

template <typename T>
DYND_CUDA_HOST_DEVICE inline complex<T> sqrt(complex<T> z)
{
  using namespace std;
  // We risk spurious overflow for components >= DBL_MAX / (1 + sqrt(2))
  const T thresh =
      (std::numeric_limits<T>::max)() / (1 + ::sqrt(static_cast<T>(2)));

  complex<T> result;
  T a = z.real(), b = z.imag();
  T t;
  bool scale;

  // Handle special cases.
  if (a == 0 && b == 0) {
    return complex<T>(0, b);
  }
  if (isinf(b)) {
    return complex<T>(std::numeric_limits<T>::infinity(), b);
  }
  if (isnan(a)) {
    t = (b - b) / (b - b);   // raise invalid if b is not a NaN
    return complex<T>(a, t); // return NaN + NaN i
  }
  if (isinf(a)) {
    // csqrt(inf + NaN i) = inf + NaN i
    // csqrt(inf + y i) = inf + 0 i
    // csqrt(-inf + NaN i) = NaN +- inf i
    // csqrt(-inf + y i) = 0 + inf i
    if (signbit(a)) {
      return complex<T>(std::fabs(b - b), copysign(a, b));
    } else {
      return complex<T>(a, copysign(b - b, b));
    }
  }
  // The remaining special case (b is NaN) is handled below

  // Scale to avoid overflow
  if (std::fabs(a) >= thresh || std::fabs(b) >= thresh) {
    a *= 0.25;
    b *= 0.25;
    scale = true;
  } else {
    scale = false;
  }

  // Algorithm 312, CACM vol 10, Oct 1967
  if (a >= 0) {
    t = std::sqrt((a + hypot(a, b)) * 0.5);
    result = complex<T>(t, b / (2 * t));
  } else {
    t = std::sqrt((-a + hypot(a, b)) * 0.5);
    result = complex<T>(std::fabs(b) / (2 * t), copysign(t, b));
  }

  // Rescale
  if (scale) {
    return complex<T>(result.real() * 2, result.imag());
  } else {
    return result;
  }
}

template <typename T>
DYND_CUDA_HOST_DEVICE complex<T> pow(complex<T> x, complex<T> y)
{
  T yr = y.real(), yi = y.imag();

  complex<T> b = log(x);
  T br = b.real(), bi = b.imag();

  return exp(complex<T>(br * yr - bi * yi, br * yi + bi * yr));
}

template <typename T>
DYND_CUDA_HOST_DEVICE complex<T> pow(complex<T> x, T y)
{
  complex<T> b = log(x);
  T br = b.real(), bi = b.imag();

  return exp(complex<T>(br * y, bi * y));
}

template <typename T>
DYND_CUDA_HOST_DEVICE complex<T> cos(complex<T> z)
{
  T x = z.real(), y = z.imag();
  return complex<T>(cos(x) * cosh(y), -(sin(x) * sinh(y)));
}

template <typename T>
DYND_CUDA_HOST_DEVICE complex<T> sin(complex<T> z)
{
  T x = z.real(), y = z.imag();
  return complex<T>(sin(x) * cosh(y), cos(x) * sinh(y));
}

} // namespace dynd
