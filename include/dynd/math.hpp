//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/types/complex.hpp>

namespace dynd {

template <typename T>
T _e(); // e

template <typename T>
T _log2_e(); // log2(e)

template <typename T>
T _log10_e(); // log10(e)

template <typename T>
T _log_2(); // log(2)

template <typename T>
T _log_10(); // log(10)

template <typename T>
T _pi(); // pi

template <typename T>
T _2_pi(); // 2 * pi

template <typename T>
T _pi_by_2(); // pi / 2

template <typename T>
T _pi_by_4(); // pi / 4

template <typename T>
T _1_by_pi(); // 1 / pi

template <typename T>
T _2_by_pi(); // 2 / pi

template <typename T>
T _sqrt_2(); // sqrt(2)

template <typename T>
T _1_by_sqrt_2(); // 1 / sqrt(2)

template <typename T>
complex<T> _i(); // complex<T>(0, 1)

template <>
inline float _e<float>()
{
  return 2.718281828459045235360287471352662498f;
}

template <>
inline float _log2_e<float>()
{
  return 1.442695040888963407359924681001892137f;
}

template <>
inline float _log10_e<float>()
{
  return 0.434294481903251827651128918916605082f;
}

template <>
inline float _log_2<float>()
{
  return 0.693147180559945309417232121458176568f;
}

template <>
inline float _log_10<float>()
{
  return 2.302585092994045684017991454684364208f;
}

template <>
inline float _pi<float>()
{
  return 3.141592653589793238462643383279502884f;
}

template <>
inline float _2_pi<float>()
{
  return 6.283185307179586231995926937088370323f;
}

template <>
inline float _pi_by_2<float>()
{
  return 1.570796326794896619231321691639751442f;
}

template <>
inline float _pi_by_4<float>()
{
  return 0.785398163397448309615660845819875721f;
}

template <>
inline float _1_by_pi<float>()
{
  return 0.318309886183790671537767526745028724f;
}

template <>
inline float _2_by_pi<float>()
{
  return 0.636619772367581343075535053490057448f;
}

template <>
inline float _sqrt_2<float>()
{
  return 1.414213562373095048801688724209698079f;
}

template <>
inline float _1_by_sqrt_2<float>()
{
  return 0.707106781186547524400844362104849039f;
}

template <>
inline complex<float> _i<float>()
{
  return complex<float>(0.0f, 1.0f);
}

template <>
inline double _e<double>()
{
  return 2.718281828459045235360287471352662498;
}

template <>
inline double _log2_e<double>()
{
  return 1.442695040888963407359924681001892137;
}

template <>
inline double _log10_e<double>()
{
  return 0.434294481903251827651128918916605082;
}

template <>
inline double _log_2<double>()
{
  return 0.693147180559945309417232121458176568;
}

template <>
inline double _log_10<double>()
{
  return 2.302585092994045684017991454684364208;
}

template <>
inline double _pi<double>()
{
  return 3.141592653589793238462643383279502884;
}

template <>
inline double _2_pi<double>()
{
  return 6.283185307179586231995926937088370323;
}

template <>
inline double _pi_by_2<double>()
{
  return 1.570796326794896619231321691639751442;
}

template <>
inline double _pi_by_4<double>()
{
  return 0.785398163397448309615660845819875721;
}

template <>
inline double _1_by_pi<double>()
{
  return 0.318309886183790671537767526745028724;
}

template <>
inline double _2_by_pi<double>()
{
  return 0.636619772367581343075535053490057448;
}

template <>
inline double _sqrt_2<double>()
{
  return 1.414213562373095048801688724209698079;
}

template <>
inline double _1_by_sqrt_2<double>()
{
  return 0.707106781186547524400844362104849039;
}

template <>
inline complex<double> _i<double>()
{
  return complex<double>(0.0, 1.0);
}

} // namespace dynd