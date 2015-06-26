//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

// This file is an internal implementation detail of built-in value assignment
// for aligned values in native byte order.

#include <dynd/fpstatus.hpp>
#include <cmath>
#include <complex>
#include <limits>

#include <dynd/config.hpp>
#include <dynd/type.hpp>
#include <dynd/diagnostics.hpp>

#if defined(_MSC_VER)
// Tell the visual studio compiler we're accessing the FPU flags
#pragma fenv_access(on)
#endif

namespace std {

template <>
struct is_signed<dynd::int128> {
  static const bool value = true;
};

template <>
struct is_signed<dynd::uint128> {
  static const bool value = false;
};

template <>
struct is_unsigned<dynd::int128> {
  static const bool value = false;
};

template <>
struct is_unsigned<dynd::uint128> {
  static const bool value = true;
};
}

namespace dynd {

template <typename DstType, typename SrcType>
typename std::enable_if<(sizeof(DstType) < sizeof(SrcType)) &&
                            std::is_signed<DstType>::value &&
                            std::is_signed<SrcType>::value,
                        bool>::type
is_overflow(SrcType src)
{
  return src < static_cast<SrcType>(std::numeric_limits<DstType>::min()) ||
         src > static_cast<SrcType>(std::numeric_limits<DstType>::max());
}

template <typename DstType, typename SrcType>
typename std::enable_if<(sizeof(DstType) >= sizeof(SrcType)) &&
                            std::is_signed<DstType>::value &&
                            std::is_signed<SrcType>::value,
                        bool>::type
is_overflow(SrcType DYND_UNUSED(src))
{
  return false;
}

template <typename DstType, typename SrcType>
typename std::enable_if<(sizeof(DstType) < sizeof(SrcType)) &&
                            std::is_signed<DstType>::value &&
                            std::is_unsigned<SrcType>::value,
                        bool>::type
is_overflow(SrcType src)
{
  return src > static_cast<SrcType>(std::numeric_limits<DstType>::max());
}

template <typename DstType, typename SrcType>
typename std::enable_if<(sizeof(DstType) >= sizeof(SrcType)) &&
                            std::is_signed<DstType>::value &&
                            std::is_unsigned<SrcType>::value,
                        bool>::type
is_overflow(SrcType DYND_UNUSED(src))
{
  return false;
}

template <typename DstType, typename SrcType>
typename std::enable_if<(sizeof(DstType) < sizeof(SrcType)) &&
                            std::is_unsigned<DstType>::value &&
                            std::is_signed<SrcType>::value,
                        bool>::type
is_overflow(SrcType src)
{
  return src < static_cast<SrcType>(0) ||
         static_cast<SrcType>(std::numeric_limits<DstType>::max()) < src;
}

template <typename DstType, typename SrcType>
typename std::enable_if<(sizeof(DstType) >= sizeof(SrcType)) &&
                            std::is_unsigned<DstType>::value &&
                            std::is_signed<SrcType>::value,
                        bool>::type
is_overflow(SrcType src)
{
  return src < static_cast<SrcType>(0);
}

template <typename DstType, typename SrcType>
typename std::enable_if<(sizeof(DstType) < sizeof(SrcType)) &&
                            std::is_unsigned<DstType>::value &&
                            std::is_unsigned<SrcType>::value,
                        bool>::type
is_overflow(SrcType src)
{
  return static_cast<SrcType>(std::numeric_limits<DstType>::max()) < src;
}

template <typename DstType, typename SrcType>
typename std::enable_if<(sizeof(DstType) >= sizeof(SrcType)) &&
                            std::is_unsigned<DstType>::value &&
                            std::is_unsigned<SrcType>::value,
                        bool>::type
is_overflow(SrcType DYND_UNUSED(src))
{
  return false;
}

template <class dst_type, class src_type, assign_error_mode errmode>
struct single_assigner_builtin_base_error {
  static void assign(dst_type *DYND_UNUSED(dst),
                     const src_type *DYND_UNUSED(src))
  {
    // DYND_TRACE_ASSIGNMENT(static_cast<float>(*src), float, *src, double);

    std::stringstream ss;
    ss << "assignment from " << ndt::make_type<src_type>() << " to "
       << ndt::make_type<dst_type>();
    ss << "with error mode " << errmode << " is not implemented";
    throw std::runtime_error(ss.str());
  }
};

template <class dst_type, class src_type, type_kind_t dst_kind,
          type_kind_t src_kind, assign_error_mode errmode>
struct single_assigner_builtin_base
    : public single_assigner_builtin_base_error<dst_type, src_type, errmode> {
};

// Any assignment with no error checking
template <class dst_type, class src_type, type_kind_t dst_kind,
          type_kind_t src_kind>
struct single_assigner_builtin_base<dst_type, src_type, dst_kind, src_kind,
                                    assign_error_nocheck> {
  

  DYND_CUDA_HOST_DEVICE static void assign(dst_type *dst, const src_type *src)
  {
    DYND_TRACE_ASSIGNMENT(static_cast<dst_type>(*src), dst_type, *src,
                          src_type);

    *dst = static_cast<dst_type>(*src);
  }
};


// complex<T> -> T with overflow checking
template <typename real_type>
struct single_assigner_builtin_base<real_type, complex<real_type>, real_kind,
                                    complex_kind, assign_error_overflow> {
  static void assign(real_type *dst, const complex<real_type> *src)
  {
    complex<real_type> s = *src;

    DYND_TRACE_ASSIGNMENT(static_cast<float>(s.real()), real_type, s,
                          complex<real_type>);

    if (s.imag() != 0) {
      std::stringstream ss;
      ss << "loss of imaginary component while assigning "
         << ndt::make_type<complex<real_type>>() << " value ";
      ss << *src << " to " << ndt::make_type<real_type>();
      throw std::runtime_error(ss.str());
    }

    *dst = s.real();
  }
};

// complex<T> -> T with fractional checking
template <typename real_type>
struct single_assigner_builtin_base<real_type, complex<real_type>, real_kind,
                                    complex_kind, assign_error_fractional>
    : public single_assigner_builtin_base<real_type, complex<real_type>,
                                          real_kind, complex_kind,
                                          assign_error_overflow> {
};

// complex<T> -> T with inexact checking
template <typename real_type>
struct single_assigner_builtin_base<real_type, complex<real_type>, real_kind,
                                    complex_kind, assign_error_inexact>
    : public single_assigner_builtin_base<real_type, complex<real_type>,
                                          real_kind, complex_kind,
                                          assign_error_overflow> {
};

template <typename real_type, assign_error_mode errmode>
struct single_assigner_builtin_base<complex<real_type>, real_type, complex_kind,
                                    real_kind, errmode>
    : public single_assigner_builtin_base<complex<real_type>, real_type,
                                          complex_kind, real_kind,
                                          assign_error_nocheck> {
};

// This is the interface exposed for use outside of this file
template <class dst_type, class src_type, assign_error_mode errmode>
struct single_assigner_builtin
    : public single_assigner_builtin_base<
          dst_type, src_type, dynd_kind_of<dst_type>::value,
          dynd_kind_of<src_type>::value, errmode> {
};
template <class same_type, assign_error_mode errmode>
struct single_assigner_builtin<same_type, same_type, errmode> {
  DYND_CUDA_HOST_DEVICE static void assign(same_type *dst, const same_type *src)
  {
    *dst = *src;
  }
};

} // namespace dynd
