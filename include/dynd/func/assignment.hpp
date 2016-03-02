//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/callable.hpp>
#include <dynd/math.hpp>
#include <dynd/kernels/base_kernel.hpp>

namespace dynd {

// Signed int -> complex floating point with inexact checking
template <typename RetType, typename ArgType>
std::enable_if_t<is_complex<RetType>::value && is_signed<ArgType>::value && is_integral<ArgType>::value, RetType>
check_cast(ArgType arg)
{
  typename RetType::value_type d = static_cast<typename RetType::value_type>(arg);

  if (static_cast<ArgType>(d) != arg) {
    std::stringstream ss;
    ss << "inexact value while assigning " << ndt::make_type<ArgType>() << " value ";
    ss << arg << " to " << ndt::make_type<RetType>() << " value " << d;
    throw std::runtime_error(ss.str());
  }
  return d;
}

// Unsigned int -> floating point with inexact checking
template <typename RetType, typename ArgType>
std::enable_if_t<is_floating_point<RetType>::value && is_unsigned<ArgType>::value && is_integral<ArgType>::value,
                 RetType>
check_cast(ArgType arg)
{
  RetType res = static_cast<RetType>(arg);

  if (static_cast<ArgType>(res) != arg) {
    std::stringstream ss;
    ss << "inexact value while assigning " << ndt::make_type<ArgType>() << " value ";
    ss << arg << " to " << ndt::make_type<RetType>() << " value " << res;
    throw std::runtime_error(ss.str());
  }
  return res;
}

// Unsigned int -> complex floating point with inexact checking
template <typename RetType, typename ArgType>
std::enable_if_t<is_complex<RetType>::value && is_unsigned<ArgType>::value && is_integral<ArgType>::value, RetType>
check_cast(ArgType arg)
{
  typename RetType::value_type res = static_cast<typename RetType::value_type>(arg);

  if (static_cast<ArgType>(res) != arg) {
    std::stringstream ss;
    ss << "inexact value while assigning " << ndt::make_type<ArgType>() << " value ";
    ss << arg << " to " << ndt::make_type<RetType>() << " value " << res;
    throw std::runtime_error(ss.str());
  }
  return res;
}

// real -> real with inexact checking
template <typename RetType, typename ArgType>
std::enable_if_t<is_floating_point<RetType>::value && is_floating_point<ArgType>::value, RetType>
check_cast(ArgType arg)
{
  RetType res;
#if defined(DYND_USE_FPSTATUS)
  clear_fp_status();
  res = static_cast<RetType>(arg);
  if (is_overflow_fp_status()) {
    std::stringstream ss;
    ss << "overflow while assigning " << ndt::make_type<ArgType>() << " value ";
    ss << arg << " to " << ndt::make_type<RetType>();
    throw std::overflow_error(ss.str());
  }
#else
  if (isfinite(arg) && (arg < -std::numeric_limits<RetType>::max() || arg > std::numeric_limits<RetType>::max())) {
    std::stringstream ss;
    ss << "overflow while assigning " << ndt::make_type<ArgType>() << " value ";
    ss << arg << " to " << ndt::make_type<RetType>();
    throw std::runtime_error(ss.str());
  }
  res = static_cast<RetType>(arg);
#endif // DYND_USE_FPSTATUS

  // The inexact status didn't work as it should have, so converting back
  // to
  // double and comparing
  // if (is_inexact_fp_status()) {
  //    throw std::runtime_error("inexact precision loss while assigning
  //    double to float");
  //}
  if (res != arg) {
    std::stringstream ss;
    ss << "inexact precision loss while assigning " << ndt::make_type<ArgType>() << " value ";
    ss << arg << " to " << ndt::make_type<RetType>();
    throw std::runtime_error(ss.str());
  }
  return res;
}

// complex<double> -> float with inexact checking
template <typename RetType, typename ArgType>
std::enable_if_t<std::is_same<RetType, float>::value && std::is_same<ArgType, complex<double>>::value, RetType>
check_cast(ArgType arg)
{
  float res;

  if (arg.imag() != 0) {
    std::stringstream ss;
    ss << "loss of imaginary component while assigning " << ndt::make_type<complex<double>>() << " value ";
    ss << arg << " to " << ndt::make_type<float>();
    throw std::runtime_error(ss.str());
  }

#if defined(DYND_USE_FPSTATUS)
  clear_fp_status();
  res = static_cast<float>(arg.real());
  if (is_overflow_fp_status()) {
    std::stringstream ss;
    ss << "overflow while assigning " << ndt::make_type<complex<double>>() << " value ";
    ss << arg << " to " << ndt::make_type<float>();
    throw std::overflow_error(ss.str());
  }
#else
  if (arg.real() < -std::numeric_limits<float>::max() || arg.real() > std::numeric_limits<float>::max()) {
    std::stringstream ss;
    ss << "overflow while assigning " << ndt::make_type<complex<double>>() << " value ";
    ss << arg << " to " << ndt::make_type<float>();
    throw std::overflow_error(ss.str());
  }
  res = static_cast<float>(arg.real());
#endif // DYND_USE_FPSTATUS

  if (res != arg.real()) {
    std::stringstream ss;
    ss << "inexact precision loss while assigning " << ndt::make_type<complex<double>>() << " value ";
    ss << arg << " to " << ndt::make_type<float>();
    throw std::runtime_error(ss.str());
  }
  return res;
}

// real -> complex with inexact checking
template <typename RetType, typename ArgType>
std::enable_if_t<is_complex<RetType>::value && is_floating_point<ArgType>::value, RetType> check_cast(ArgType s)
{
  typename RetType::value_type d;

#if defined(DYND_USE_FPSTATUS)
  clear_fp_status();
  d = static_cast<typename RetType::value_type>(s);
  if (is_overflow_fp_status()) {
    std::stringstream ss;
    ss << "overflow while assigning " << ndt::make_type<ArgType>() << " value ";
    ss << s << " to " << ndt::make_type<RetType>();
    throw std::overflow_error(ss.str());
  }
#else
  if (isfinite(s) && (s < -std::numeric_limits<typename RetType::value_type>::max() ||
                      s > std::numeric_limits<typename RetType::value_type>::max())) {
    std::stringstream ss;
    ss << "overflow while assigning " << ndt::make_type<ArgType>() << " value ";
    ss << s << " to " << ndt::make_type<RetType>();
    throw std::overflow_error(ss.str());
  }
  d = static_cast<typename RetType::value_type>(s);
#endif // DYND_USE_FPSTATUS

  if (d != s) {
    std::stringstream ss;
    ss << "inexact precision loss while assigning " << ndt::make_type<ArgType>() << " value ";
    ss << s << " to " << ndt::make_type<RetType>();
    throw std::runtime_error(ss.str());
  }
  return d;
}

// complex<double> -> complex<float> with inexact checking
template <typename RetType, typename ArgType>
std::enable_if_t<std::is_same<RetType, complex<float>>::value && std::is_same<ArgType, complex<double>>::value, RetType>
check_cast(ArgType s)
{
  complex<float> d;

#if defined(DYND_USE_FPSTATUS)
  clear_fp_status();
  d = static_cast<complex<float>>(s);
  if (is_overflow_fp_status()) {
    std::stringstream ss;
    ss << "overflow while assigning " << ndt::make_type<complex<double>>() << " value ";
    ss << s << " to " << ndt::make_type<complex<float>>();
    throw std::overflow_error(ss.str());
  }
#else
  if (s.real() < -std::numeric_limits<float>::max() || s.real() > std::numeric_limits<float>::max() ||
      s.imag() < -std::numeric_limits<float>::max() || s.imag() > std::numeric_limits<float>::max()) {
    std::stringstream ss;
    ss << "overflow while assigning " << ndt::make_type<complex<double>>() << " value ";
    ss << s << " to " << ndt::make_type<complex<float>>();
    throw std::overflow_error(ss.str());
  }
  d = static_cast<complex<float>>(s);
#endif // DYND_USE_FPSTATUS

  // The inexact status didn't work as it should have, so converting back
  // to
  // double and comparing
  // if (is_inexact_fp_status()) {
  //    throw std::runtime_error("inexact precision loss while assigning
  //    double to float");
  //}
  if (d.real() != s.real() || d.imag() != s.imag()) {
    std::stringstream ss;
    ss << "inexact precision loss while assigning " << ndt::make_type<complex<double>>() << " value ";
    ss << s << " to " << ndt::make_type<complex<float>>();
    throw std::runtime_error(ss.str());
  }
  return d;
}

// Signed int -> floating point with inexact checking
template <typename RetType, typename ArgType>
std::enable_if_t<is_floating_point<RetType>::value && is_signed<ArgType>::value && is_integral<ArgType>::value, RetType>
check_cast(ArgType s)
{
  RetType d = static_cast<RetType>(s);

  if (static_cast<ArgType>(d) != s) {
    std::stringstream ss;
    ss << "inexact value while assigning " << ndt::make_type<ArgType>() << " value ";
    ss << s << " to " << ndt::make_type<RetType>() << " value " << d;
    throw std::runtime_error(ss.str());
  }

  return d;
}

namespace nd {

  extern DYND_API struct DYND_API assign : declfunc<assign> {
    static callable make();
    static callable &get();
  } assign;

} // namespace dynd::nd

/**
 * Creates an assignment kernel for one data value from the
 * src type/arrmeta to the dst type/arrmeta. This adds the
 * kernel at the 'ckb_offset' position in 'ckb's data, as part
 * of a hierarchy matching the dynd type's hierarchy.
 *
 * This function should always be called with this == dst_tp first,
 * and types which don't support the particular assignment should
 * then call the corresponding function with this == src_dt.
 *
 * \param ckb  The ckernel_builder being constructed.
 * \param ckb_offset  The offset within 'ckb'.
 * \param dst_tp  The destination dynd type.
 * \param dst_arrmeta  Arrmeta for the destination data.
 * \param src_tp  The source dynd type.
 * \param src_arrmeta  Arrmeta for the source data
 * \param kernreq  What kind of kernel must be placed in 'ckb'.
 * \param ectx  DyND evaluation context.
 *
 * \returns  The offset within 'ckb' immediately after the
 *           created kernel.
 */
DYND_API void make_assignment_kernel(nd::kernel_builder *ckb, const ndt::type &dst_tp, const char *dst_arrmeta,
                                     const ndt::type &src_tp, const char *src_arrmeta, kernel_request_t kernreq,
                                     const eval::eval_context *ectx);

inline void make_assignment_kernel(nd::kernel_builder *ckb, const ndt::type &dst_tp, const char *dst_arrmeta,
                                   const ndt::type *src_tp, const char *const *src_arrmeta, kernel_request_t kernreq,
                                   const eval::eval_context *ectx)
{
  make_assignment_kernel(ckb, dst_tp, dst_arrmeta, *src_tp, *src_arrmeta, kernreq, ectx);
}

/**
 * Creates an assignment kernel when the src and the dst are the same,
 * and are POD (plain old data).
 *
 * \param ckb  The hierarchical assignment kernel being constructed.
 * \param ckb_offset  The offset within 'ckb'.
 * \param data_size  The size of the data being assigned.
 * \param data_alignment  The alignment of the data being assigned.
 * \param kernreq  What kind of kernel must be placed in 'ckb'.
 */
DYND_API void make_pod_typed_data_assignment_kernel(nd::kernel_builder *ckb, size_t data_size, size_t data_alignment,
                                                    kernel_request_t kernreq);

} // namespace dynd
