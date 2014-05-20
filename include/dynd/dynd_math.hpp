//
// Copyright (C) 2011-14 Irwin Zaid, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DYND__DYND_MATH_HPP_
#define _DYND__DYND_MATH_HPP_

#include <dynd/dynd_math.h>

namespace dynd {

template <typename T>
T dynd_e();

template <typename T>
T dynd_log2_e();

template <typename T>
T dynd_log10_e();

template <typename T>
T dynd_log_2();

template <typename T>
T dynd_log_10();

template <typename T>
T dynd_pi();

template <typename T>
T dynd_2_mul_pi();

template <typename T>
T dynd_pi_div_2();

template <typename T>
T dynd_pi_div_4();

template <typename T>
T dynd_1_div_pi();

template <typename T>
T dynd_2_div_pi();

template <typename T>
T dynd_sqrt_pi();

template <typename T>
T dynd_1_div_sqrt_2();

template <>
inline float dynd_e<float>() {
    return DYND_EF;
}

template <>
inline float dynd_log2_e<float>() {
    return DYND_LOG2_EF;
}

template <>
inline float dynd_log10_e<float>() {
    return DYND_LOG10_EF;
}

template <>
inline float dynd_log_2<float>() {
    return DYND_LOG_2F;
}

template <>
inline float dynd_log_10<float>() {
    return DYND_LOG_10F;
}

template <>
inline float dynd_pi<float>() {
    return DYND_PIF;
}

template <>
inline float dynd_2_mul_pi<float>() {
    return DYND_2F_MUL_PIF;
}

template <>
inline float dynd_pi_div_2<float>() {
    return DYND_PIF_DIV_2F;
}

template <>
inline float dynd_pi_div_4<float>() {
    return DYND_PIF_DIV_4F;
}

template <>
inline float dynd_1_div_pi<float>() {
    return DYND_1F_DIV_PIF;
}

template <>
inline float dynd_2_div_pi<float>() {
    return DYND_2F_DIV_PIF;
}

template <>
inline float dynd_sqrt_pi<float>() {
    return DYND_SQRT_PIF;
}

template <>
inline float dynd_1_div_sqrt_2<float>() {
    return DYND_1F_DIV_SQRT_2F;
}

template <>
inline double dynd_e<double>() {
    return DYND_E;
}

template <>
inline double dynd_log2_e<double>() {
    return DYND_LOG2_E;
}

template <>
inline double dynd_log10_e<double>() {
    return DYND_LOG10_E;
}

template <>
inline double dynd_log_2<double>() {
    return DYND_LOG_2;
}

template <>
inline double dynd_log_10<double>() {
    return DYND_LOG_10;
}

template <>
inline double dynd_pi<double>() {
    return DYND_PI;
}

template <>
inline double dynd_2_mul_pi<double>() {
    return DYND_2_MUL_PI;
}

template <>
inline double dynd_pi_div_2<double>() {
    return DYND_PI_DIV_2;
}

template <>
inline double dynd_pi_div_4<double>() {
    return DYND_PI_DIV_4;
}

template <>
inline double dynd_1_div_pi<double>() {
    return DYND_1_DIV_PI;
}

template <>
inline double dynd_2_div_pi<double>() {
    return DYND_2_DIV_PI;
}

template <>
inline double dynd_sqrt_pi<double>() {
    return DYND_SQRT_PI;
}

template <>
inline double dynd_1_div_sqrt_2<double>() {
    return DYND_1_DIV_SQRT_2;
}

} // namespace dynd

#endif
