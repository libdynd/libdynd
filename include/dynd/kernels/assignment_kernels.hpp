//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <stdexcept>

#include <dynd/type.hpp>
#include <dynd/typed_data_assign.hpp>
#include <dynd/func/assignment.hpp>
#include <dynd/kernels/cuda_launch.hpp>
#include <dynd/kernels/base_kernel.hpp>
#include <dynd/eval/eval_context.hpp>
#include <dynd/typed_data_assign.hpp>
#include <dynd/types/type_id.hpp>
#include <dynd/kernels/single_assigner_builtin.hpp>
#include <map>

namespace dynd {

namespace nd {

  template <type_id_t DstTypeID, type_kind_t DstTypeKind, type_id_t Src0TypeID,
            type_kind_t Src0TypeKind, assign_error_mode ErrorMode>
  struct assignment_kernel
      : base_kernel<assignment_kernel<DstTypeID, DstTypeKind, Src0TypeID,
                                      Src0TypeKind, ErrorMode>,
                    kernel_request_host, 1> {
    typedef typename type_of<DstTypeID>::type dst_type;
    typedef typename type_of<Src0TypeID>::type src_type;

    void single(char *dst, char *const *src)
    {
      single_assigner_builtin<dst_type, src_type, ErrorMode>::assign(
          reinterpret_cast<dst_type *>(dst),
          reinterpret_cast<src_type *>(*src));
    }
  };

  /*
    template <type_id_t DstTypeID, type_kind_t DstTypeKind, type_id_t
  Src0TypeID,
              type_kind_t Src0TypeKind>
    struct assignment_kernel<DstTypeID, DstTypeKind, Src0TypeID, Src0TypeKind,
                             assign_error_nocheck>
        : base_kernel<assignment_kernel<DstTypeID, DstTypeKind, Src0TypeID,
                                        Src0TypeKind, assign_error_nocheck>,
                      kernel_request_host, 1> {
      typedef typename type_of<DstTypeID>::type dst_type;
      typedef typename type_of<Src0TypeID>::type src0_type;

      void single(char *DYND_UNUSED(dst), char *const *DYND_UNUSED(src))
      {
  // DYND_TRACE_ASSIGNMENT(static_cast<float>(*src), float, *src, double);

  #ifdef __CUDA_ARCH__
        DYND_TRIGGER_ASSERT(
            "assignment is not implemented for CUDA global memory");
  #else
        std::stringstream ss;
        ss << "assignment from " << ndt::make_type<src0_type>() << " to "
           << ndt::make_type<dst_type>();
        ss << "with error mode " << assign_error_nocheck << " is not
  implemented";
        throw std::runtime_error(ss.str());
  #endif
      }
    };
  */

  // Anything -> boolean with no checking
  template <type_id_t Src0TypeID, type_kind_t Src0TypeKind>
  struct assignment_kernel<bool_type_id, bool_kind, Src0TypeID, Src0TypeKind,
                           assign_error_nocheck>
      : base_kernel<assignment_kernel<bool_type_id, bool_kind, Src0TypeID,
                                      Src0TypeKind, assign_error_nocheck>,
                    kernel_request_host, 1> {
    typedef typename type_of<Src0TypeID>::type src0_type;

    void single(char *dst, char *const *src)
    {
      src0_type s = *reinterpret_cast<src0_type *>(src[0]);

      DYND_TRACE_ASSIGNMENT((bool)(s != src0_type(0)), bool1, s, src0_type);

      *reinterpret_cast<bool1 *>(dst) = (s != src0_type(0));
    }
  };

  // Anything -> boolean with overflow checking
  template <type_id_t Src0TypeID, type_kind_t Src0TypeKind>
  struct assignment_kernel<bool_type_id, bool_kind, Src0TypeID, Src0TypeKind,
                           assign_error_overflow>
      : base_kernel<assignment_kernel<bool_type_id, bool_kind, Src0TypeID,
                                      Src0TypeKind, assign_error_overflow>,
                    kernel_request_host, 1> {
    typedef typename type_of<Src0TypeID>::type src0_type;

    void single(char *dst, char *const *src)
    {
      src0_type s = *reinterpret_cast<src0_type *>(src[0]);

      DYND_TRACE_ASSIGNMENT((bool)(s != src0_type(0)), bool1, s, src0_type);

      if (s == src0_type(0)) {
        *dst = false;
      } else if (s == src0_type(1)) {
        *dst = true;
      } else {
        std::stringstream ss;
        ss << "overflow while assigning " << ndt::make_type<src0_type>()
           << " value ";
        ss << s << " to " << ndt::make_type<bool1>();
        throw std::overflow_error(ss.str());
      }
    }
  };

  // Anything -> boolean with other error checking
  template <type_id_t Src0TypeID, type_kind_t Src0TypeKind>
  struct assignment_kernel<bool_type_id, bool_kind, Src0TypeID, Src0TypeKind,
                           assign_error_fractional>
      : assignment_kernel<bool_type_id, bool_kind, Src0TypeID, Src0TypeKind,
                          assign_error_overflow> {
  };

  template <type_id_t Src0TypeID, type_kind_t Src0TypeKind>
  struct assignment_kernel<bool_type_id, bool_kind, Src0TypeID, Src0TypeKind,
                           assign_error_inexact>
      : assignment_kernel<bool_type_id, bool_kind, Src0TypeID, Src0TypeKind,
                          assign_error_overflow> {
  };

  // Boolean -> boolean with other error checking
  template <>
  struct assignment_kernel<bool_type_id, bool_kind, bool_type_id, bool_kind,
                           assign_error_overflow>
      : assignment_kernel<bool_type_id, bool_kind, bool_type_id, bool_kind,
                          assign_error_nocheck> {
  };

  template <>
  struct assignment_kernel<bool_type_id, bool_kind, bool_type_id, bool_kind,
                           assign_error_fractional>
      : assignment_kernel<bool_type_id, bool_kind, bool_type_id, bool_kind,
                          assign_error_nocheck> {
  };

  template <>
  struct assignment_kernel<bool_type_id, bool_kind, bool_type_id, bool_kind,
                           assign_error_inexact>
      : assignment_kernel<bool_type_id, bool_kind, bool_type_id, bool_kind,
                          assign_error_nocheck> {
  };

  // Boolean -> anything with other error checking
  template <type_id_t DstTypeID, type_kind_t DstTypeKind>
  struct assignment_kernel<DstTypeID, DstTypeKind, bool_type_id, bool_kind,
                           assign_error_overflow>
      : assignment_kernel<DstTypeID, DstTypeKind, bool_type_id, bool_kind,
                          assign_error_nocheck> {
  };

  template <type_id_t DstTypeID, type_kind_t DstTypeKind>
  struct assignment_kernel<DstTypeID, DstTypeKind, bool_type_id, bool_kind,
                           assign_error_fractional>
      : assignment_kernel<DstTypeID, DstTypeKind, bool_type_id, bool_kind,
                          assign_error_nocheck> {
  };

  template <type_id_t DstTypeID, type_kind_t DstTypeKind>
  struct assignment_kernel<DstTypeID, DstTypeKind, bool_type_id, bool_kind,
                           assign_error_inexact>
      : assignment_kernel<DstTypeID, DstTypeKind, bool_type_id, bool_kind,
                          assign_error_nocheck> {
  };

  // Signed int -> signed int with overflow checking
  template <type_id_t DstTypeID, type_id_t Src0TypeID>
  struct assignment_kernel<DstTypeID, sint_kind, Src0TypeID, sint_kind,
                           assign_error_overflow>
      : base_kernel<assignment_kernel<DstTypeID, sint_kind, Src0TypeID,
                                      sint_kind, assign_error_overflow>,
                    kernel_request_host, 1> {
    typedef typename type_of<DstTypeID>::type dst_type;
    typedef typename type_of<Src0TypeID>::type src0_type;

    void single(char *dst, char *const *src)
    {
      src0_type s = *reinterpret_cast<src0_type *>(src[0]);

      if (is_overflow<dst_type>(s)) {
        std::stringstream ss;
        ss << "overflow while assigning " << ndt::make_type<src0_type>()
           << " value ";
        ss << s << " to " << ndt::make_type<dst_type>();
        throw std::overflow_error(ss.str());
      }
      *reinterpret_cast<dst_type *>(dst) = static_cast<dst_type>(s);
    }
  };

  // Signed int -> signed int with other error checking
  template <type_id_t DstTypeID, type_id_t Src0TypeID>
  struct assignment_kernel<DstTypeID, sint_kind, Src0TypeID, sint_kind,
                           assign_error_fractional>
      : assignment_kernel<DstTypeID, sint_kind, Src0TypeID, sint_kind,
                          assign_error_overflow> {
  };

  template <type_id_t DstTypeID, type_id_t Src0TypeID>
  struct assignment_kernel<DstTypeID, sint_kind, Src0TypeID, sint_kind,
                           assign_error_inexact>
      : assignment_kernel<DstTypeID, sint_kind, Src0TypeID, sint_kind,
                          assign_error_overflow> {
  };

  // Unsigned int -> signed int with overflow checking just when sizeof(dst) <=
  // sizeof(src)
  template <type_id_t DstTypeID, type_id_t Src0TypeID>
  struct assignment_kernel<DstTypeID, sint_kind, Src0TypeID, uint_kind,
                           assign_error_overflow>
      : base_kernel<assignment_kernel<DstTypeID, sint_kind, Src0TypeID,
                                      uint_kind, assign_error_overflow>,
                    kernel_request_host, 1> {
    typedef typename type_of<DstTypeID>::type dst_type;
    typedef typename type_of<Src0TypeID>::type src0_type;

    void single(char *dst, char *const *src)
    {
      src0_type s = *reinterpret_cast<src0_type *>(src[0]);

      DYND_TRACE_ASSIGNMENT(static_cast<dst_type>(s), dst_type, s, src0_type);

      if (is_overflow<dst_type>(s)) {
        std::stringstream ss;
        ss << "overflow while assigning " << ndt::make_type<src0_type>()
           << " value ";
        ss << s << " to " << ndt::make_type<dst_type>();
        throw std::overflow_error(ss.str());
      }
      *reinterpret_cast<dst_type *>(dst) = static_cast<dst_type>(s);
    }
  };

  // Unsigned int -> signed int with other error checking
  template <type_id_t DstTypeID, type_id_t Src0TypeID>
  struct assignment_kernel<DstTypeID, sint_kind, Src0TypeID, uint_kind,
                           assign_error_fractional>
      : assignment_kernel<DstTypeID, sint_kind, Src0TypeID, uint_kind,
                          assign_error_overflow> {
  };

  template <type_id_t DstTypeID, type_id_t Src0TypeID>
  struct assignment_kernel<DstTypeID, sint_kind, Src0TypeID, uint_kind,
                           assign_error_inexact>
      : assignment_kernel<DstTypeID, sint_kind, Src0TypeID, uint_kind,
                          assign_error_overflow> {
  };

  // Signed int -> unsigned int with positive overflow checking just when
  // sizeof(dst) < sizeof(src)
  template <type_id_t DstTypeID, type_id_t Src0TypeID>
  struct assignment_kernel<DstTypeID, uint_kind, Src0TypeID, sint_kind,
                           assign_error_overflow>
      : base_kernel<assignment_kernel<DstTypeID, uint_kind, Src0TypeID,
                                      sint_kind, assign_error_overflow>,
                    kernel_request_host, 1> {
    typedef typename type_of<DstTypeID>::type dst_type;
    typedef typename type_of<Src0TypeID>::type src0_type;

    void single(char *dst, char *const *src)
    {
      src0_type s = *reinterpret_cast<src0_type *>(src[0]);

      DYND_TRACE_ASSIGNMENT(static_cast<dst_type>(s), dst_type, s, src_type);

      if (is_overflow<dst_type>(s)) {
        std::stringstream ss;
        ss << "overflow while assigning " << ndt::make_type<src0_type>()
           << " value ";
        ss << s << " to " << ndt::make_type<dst_type>();
        throw std::overflow_error(ss.str());
      }
      *reinterpret_cast<dst_type *>(dst) = static_cast<dst_type>(s);
    }
  };

  // Signed int -> unsigned int with other error checking
  template <type_id_t DstTypeID, type_id_t Src0TypeID>
  struct assignment_kernel<DstTypeID, uint_kind, Src0TypeID, sint_kind,
                           assign_error_fractional>
      : assignment_kernel<DstTypeID, uint_kind, Src0TypeID, sint_kind,
                          assign_error_overflow> {
  };

  template <type_id_t DstTypeID, type_id_t Src0TypeID>
  struct assignment_kernel<DstTypeID, uint_kind, Src0TypeID, sint_kind,
                           assign_error_inexact>
      : assignment_kernel<DstTypeID, uint_kind, Src0TypeID, sint_kind,
                          assign_error_overflow> {
  };

  // Unsigned int -> unsigned int with overflow checking just when sizeof(dst) <
  // sizeof(src)
  template <type_id_t DstTypeID, type_id_t Src0TypeID>
  struct assignment_kernel<DstTypeID, uint_kind, Src0TypeID, uint_kind,
                           assign_error_overflow>
      : base_kernel<assignment_kernel<DstTypeID, uint_kind, Src0TypeID,
                                      uint_kind, assign_error_overflow>,
                    kernel_request_host, 1> {
    typedef typename type_of<DstTypeID>::type dst_type;
    typedef typename type_of<Src0TypeID>::type src0_type;

    void single(char *dst, char *const *src)
    {
      src0_type s = *reinterpret_cast<src0_type *>(src[0]);

      DYND_TRACE_ASSIGNMENT(static_cast<dst_type>(s), dst_type, s, src_type);

      if (is_overflow<dst_type>(s)) {
        std::stringstream ss;
        ss << "overflow while assigning " << ndt::make_type<src0_type>()
           << " value ";
        ss << s << " to " << ndt::make_type<dst_type>();
        throw std::overflow_error(ss.str());
      }
      *reinterpret_cast<dst_type *>(dst) = static_cast<dst_type>(s);
    }
  };

  // Unsigned int -> unsigned int with other error checking
  template <type_id_t DstTypeID, type_id_t Src0TypeID>
  struct assignment_kernel<DstTypeID, uint_kind, Src0TypeID, uint_kind,
                           assign_error_fractional>
      : assignment_kernel<DstTypeID, uint_kind, Src0TypeID, uint_kind,
                          assign_error_overflow> {
  };

  template <type_id_t DstTypeID, type_id_t Src0TypeID>
  struct assignment_kernel<DstTypeID, uint_kind, Src0TypeID, uint_kind,
                           assign_error_inexact>
      : assignment_kernel<DstTypeID, uint_kind, Src0TypeID, uint_kind,
                          assign_error_overflow> {
  };

  // Signed int -> floating point with inexact checking
  template <type_id_t DstTypeID, type_id_t Src0TypeID>
  struct assignment_kernel<DstTypeID, real_kind, Src0TypeID, sint_kind,
                           assign_error_inexact>
      : base_kernel<assignment_kernel<DstTypeID, real_kind, Src0TypeID,
                                      sint_kind, assign_error_inexact>,
                    kernel_request_host, 1> {
    typedef typename type_of<DstTypeID>::type dst_type;
    typedef typename type_of<Src0TypeID>::type src0_type;

    void single(char *dst, char *const *src)
    {
      src0_type s = *reinterpret_cast<src0_type *>(src[0]);
      dst_type d = static_cast<dst_type>(s);

      DYND_TRACE_ASSIGNMENT(d, dst_type, s, src_type);

      if (static_cast<src0_type>(d) != s) {
        std::stringstream ss;
        ss << "inexact value while assigning " << ndt::make_type<src0_type>()
           << " value ";
        ss << s << " to " << ndt::make_type<dst_type>() << " value " << d;
        throw std::runtime_error(ss.str());
      }
      *reinterpret_cast<dst_type *>(dst) = d;
    }
  };

  // Signed int -> floating point with other checking
  template <type_id_t DstTypeID, type_id_t Src0TypeID>
  struct assignment_kernel<DstTypeID, real_kind, Src0TypeID, sint_kind,
                           assign_error_overflow>
      : assignment_kernel<DstTypeID, real_kind, Src0TypeID, sint_kind,
                          assign_error_nocheck> {
  };

  template <type_id_t DstTypeID, type_id_t Src0TypeID>
  struct assignment_kernel<DstTypeID, real_kind, Src0TypeID, sint_kind,
                           assign_error_fractional>
      : assignment_kernel<DstTypeID, real_kind, Src0TypeID, sint_kind,
                          assign_error_nocheck> {
  };

  // double -> complex<float>
  template <>
  struct assignment_kernel<complex_float32_type_id, complex_kind,
                           float64_type_id, real_kind, assign_error_nocheck>
      : base_kernel<
            assignment_kernel<complex_float32_type_id, complex_kind,
                              float64_type_id, real_kind, assign_error_nocheck>,
            kernel_request_host, 1> {
    void single(char *dst, char *const *src)
    {
      double s = *reinterpret_cast<double *>(src[0]);

      DYND_TRACE_ASSIGNMENT(static_cast<complex<float>>(s), complex<float>, s,
                            double);

      *reinterpret_cast<complex<float> *>(dst) = static_cast<float>(s);
    }
  };

  // float -> complex<double>
  template <>
  struct assignment_kernel<complex_float64_type_id, complex_kind,
                           float32_type_id, real_kind, assign_error_nocheck>
      : base_kernel<
            assignment_kernel<complex_float64_type_id, complex_kind,
                              float32_type_id, real_kind, assign_error_nocheck>,
            kernel_request_host, 1> {
    void single(char *dst, char *const *src)
    {
      float s = *reinterpret_cast<float *>(src[0]);

      DYND_TRACE_ASSIGNMENT(static_cast<complex<double>>(*src), complex<double>,
                            s, float);

      *reinterpret_cast<complex<double> *>(dst) = s;
    }
  };

  template <assign_error_mode ErrorMode>
  struct assignment_kernel<complex_float64_type_id, complex_kind,
                           float32_type_id, real_kind, ErrorMode>
      : assignment_kernel<complex_float64_type_id, complex_kind,
                          float32_type_id, real_kind, assign_error_nocheck> {
  };

  // complex<float> -> double with overflow checking
  template <>
  struct assignment_kernel<float64_type_id, real_kind, complex_float32_type_id,
                           complex_kind, assign_error_overflow>
      : base_kernel<assignment_kernel<float64_type_id, real_kind,
                                      complex_float32_type_id, complex_kind,
                                      assign_error_overflow>,
                    kernel_request_host, 1> {
    void single(char *dst, char *const *src)
    {
      complex<float> s = *reinterpret_cast<complex<float> *>(src[0]);

      DYND_TRACE_ASSIGNMENT(static_cast<double>(s.real()), double, s,
                            complex<float>);

      if (s.imag() != 0) {
        std::stringstream ss;
        ss << "loss of imaginary component while assigning "
           << ndt::make_type<complex<float>>() << " value ";
        ss << *src << " to " << ndt::make_type<double>();
        throw std::runtime_error(ss.str());
      }

      *reinterpret_cast<double *>(dst) = s.real();
    }
  };

  // complex<float> -> double with fractional checking
  template <>
  struct assignment_kernel<float64_type_id, real_kind, complex_float32_type_id,
                           complex_kind, assign_error_fractional>
      : assignment_kernel<float64_type_id, real_kind, complex_float32_type_id,
                          complex_kind, assign_error_overflow> {
  };

  // complex<float> -> double with inexact checking
  template <>
  struct assignment_kernel<float64_type_id, real_kind, complex_float32_type_id,
                           complex_kind, assign_error_inexact>
      : assignment_kernel<float64_type_id, real_kind, complex_float32_type_id,
                          complex_kind, assign_error_overflow> {
  };

  // complex<double> -> float with overflow checking
  template <>
  struct assignment_kernel<float32_type_id, real_kind, complex_float64_type_id,
                           complex_kind, assign_error_overflow>
      : base_kernel<assignment_kernel<float32_type_id, real_kind,
                                      complex_float64_type_id, complex_kind,
                                      assign_error_overflow>,
                    kernel_request_host, 1> {
    void single(char *dst, char *const *src)
    {
      complex<double> s = *reinterpret_cast<complex<double> *>(src[0]);
      float d;

      DYND_TRACE_ASSIGNMENT(static_cast<float>(s.real()), float, s,
                            complex<double>);

      if (s.imag() != 0) {
        std::stringstream ss;
        ss << "loss of imaginary component while assigning "
           << ndt::make_type<complex<double>>() << " value ";
        ss << *src << " to " << ndt::make_type<float>();
        throw std::runtime_error(ss.str());
      }

#if defined(DYND_USE_FPSTATUS)
      clear_fp_status();
      d = static_cast<float>(s.real());
      if (is_overflow_fp_status()) {
        std::stringstream ss;
        ss << "overflow while assigning " << ndt::make_type<complex<double>>()
           << " value ";
        ss << *src << " to " << ndt::make_type<float>();
        throw std::overflow_error(ss.str());
      }
#else
      if (s.real() < -std::numeric_limits<float>::max() ||
          s.real() > std::numeric_limits<float>::max()) {
        std::stringstream ss;
        ss << "overflow while assigning " << ndt::make_type<complex<double>>()
           << " value ";
        ss << *src << " to " << ndt::make_type<float>();
        throw std::overflow_error(ss.str());
      }
      d = static_cast<float>(s.real());
#endif // DYND_USE_FPSTATUS

      *reinterpret_cast<float *>(dst) = d;
    }
  };

  // complex<double> -> float with fractional checking
  template <>
  struct assignment_kernel<float32_type_id, real_kind, complex_float64_type_id,
                           complex_kind, assign_error_fractional>
      : assignment_kernel<float32_type_id, real_kind, complex_float64_type_id,
                          complex_kind, assign_error_overflow> {
  };

  // complex<double> -> float with inexact checking
  template <>
  struct assignment_kernel<float32_type_id, real_kind, complex_float64_type_id,
                           complex_kind, assign_error_inexact>
      : base_kernel<assignment_kernel<float32_type_id, real_kind,
                                      complex_float64_type_id, complex_kind,
                                      assign_error_inexact>,
                    kernel_request_host, 1> {
    void single(char *dst, char *const *src)
    {
      complex<double> s = *reinterpret_cast<complex<double> *>(src[0]);
      float d;

      DYND_TRACE_ASSIGNMENT(static_cast<float>(s.real()), float, s,
                            complex<double>);

      if (s.imag() != 0) {
        std::stringstream ss;
        ss << "loss of imaginary component while assigning "
           << ndt::make_type<complex<double>>() << " value ";
        ss << *src << " to " << ndt::make_type<float>();
        throw std::runtime_error(ss.str());
      }

#if defined(DYND_USE_FPSTATUS)
      clear_fp_status();
      d = static_cast<float>(s.real());
      if (is_overflow_fp_status()) {
        std::stringstream ss;
        ss << "overflow while assigning " << ndt::make_type<complex<double>>()
           << " value ";
        ss << s << " to " << ndt::make_type<float>();
        throw std::overflow_error(ss.str());
      }
#else
      if (s.real() < -std::numeric_limits<float>::max() ||
          s.real() > std::numeric_limits<float>::max()) {
        std::stringstream ss;
        ss << "overflow while assigning " << ndt::make_type<complex<double>>()
           << " value ";
        ss << s << " to " << ndt::make_type<float>();
        throw std::overflow_error(ss.str());
      }
      d = static_cast<float>(s.real());
#endif // DYND_USE_FPSTATUS

      if (d != s.real()) {
        std::stringstream ss;
        ss << "inexact precision loss while assigning "
           << ndt::make_type<complex<double>>() << " value ";
        ss << *src << " to " << ndt::make_type<float>();
        throw std::runtime_error(ss.str());
      }

      *reinterpret_cast<float *>(dst) = d;
    }
  };

  // double -> complex<float> with overflow checking
  template <>
  struct assignment_kernel<complex_float32_type_id, complex_kind,
                           float64_type_id, real_kind, assign_error_overflow>
      : base_kernel<assignment_kernel<complex_float32_type_id, complex_kind,
                                      float64_type_id, real_kind,
                                      assign_error_overflow>,
                    kernel_request_host, 1> {

    void single(char *dst, char *const *src)
    {
      double s = *reinterpret_cast<double *>(src[0]);
      float d;

      DYND_TRACE_ASSIGNMENT(static_cast<complex<float>>(s), complex<float>, s,
                            double);

#if defined(DYND_USE_FPSTATUS)
      clear_fp_status();
      d = static_cast<float>(s);
      if (is_overflow_fp_status()) {
        std::stringstream ss;
        ss << "overflow while assigning " << ndt::make_type<double>()
           << " value ";
        ss << s << " to " << ndt::make_type<complex<float>>();
        throw std::overflow_error(ss.str());
      }
#else
      if (isfinite(s) && (s < -std::numeric_limits<float>::max() ||
                          s > std::numeric_limits<float>::max())) {
        std::stringstream ss;
        ss << "overflow while assigning " << ndt::make_type<double>()
           << " value ";
        ss << s << " to " << ndt::make_type<complex<float>>();
        throw std::overflow_error(ss.str());
      }
      d = static_cast<float>(s);
#endif // DYND_USE_FPSTATUS

      *reinterpret_cast<complex<float> *>(dst) = d;
    }
  };

  // double -> complex<float> with fractional checking
  template <>
  struct assignment_kernel<complex_float32_type_id, complex_kind,
                           float64_type_id, real_kind, assign_error_fractional>
      : assignment_kernel<complex_float32_type_id, complex_kind,
                          float64_type_id, real_kind, assign_error_overflow> {
  };

  // double -> complex<float> with inexact checking
  template <>
  struct assignment_kernel<complex_float32_type_id, complex_kind,
                           float64_type_id, real_kind, assign_error_inexact>
      : base_kernel<
            assignment_kernel<complex_float32_type_id, complex_kind,
                              float64_type_id, real_kind, assign_error_inexact>,
            kernel_request_host, 1> {

    void single(char *dst, char *const *src)
    {
      double s = *reinterpret_cast<double *>(src[0]);
      float d;

      DYND_TRACE_ASSIGNMENT(static_cast<complex<float>>(s), complex<float>, s,
                            double);

#if defined(DYND_USE_FPSTATUS)
      clear_fp_status();
      d = static_cast<float>(s);
      if (is_overflow_fp_status()) {
        std::stringstream ss;
        ss << "overflow while assigning " << ndt::make_type<double>()
           << " value ";
        ss << s << " to " << ndt::make_type<complex<float>>();
        throw std::overflow_error(ss.str());
      }
#else
      if (isfinite(s) && (s < -std::numeric_limits<float>::max() ||
                          s > std::numeric_limits<float>::max())) {
        std::stringstream ss;
        ss << "overflow while assigning " << ndt::make_type<double>()
           << " value ";
        ss << s << " to " << ndt::make_type<complex<float>>();
        throw std::overflow_error(ss.str());
      }
      d = static_cast<float>(s);
#endif // DYND_USE_FPSTATUS

      if (d != s) {
        std::stringstream ss;
        ss << "inexact precision loss while assigning "
           << ndt::make_type<double>() << " value ";
        ss << s << " to " << ndt::make_type<complex<float>>();
        throw std::runtime_error(ss.str());
      }

      *reinterpret_cast<complex<float> *>(dst) = d;
    }
  };

/*
  template <type_class dst_type, class src_type>
  struct assign_ck<dst_type, src_type, assign_error_nocheck>
      : base_kernel<assign_ck<dst_type, src_type, assign_error_nocheck>,
                    kernel_request_cuda_host_device, 1> {
    DYND_CUDA_HOST_DEVICE void single(char *dst, char *const *src)
    {
      single_assigner_builtin<dst_type, src_type, assign_error_nocheck>::assign(
          reinterpret_cast<dst_type *>(dst),
          reinterpret_cast<src_type *>(*src));
    }
  };
*/

#ifdef DYND_CUDA

  struct cuda_host_to_device_assign_ck
      : nd::expr_ck<cuda_host_to_device_assign_ck, kernel_request_host, 1> {
    size_t data_size;
    char *dst;

    cuda_host_to_device_assign_ck(size_t data_size)
        : data_size(data_size), dst(new char[data_size])
    {
    }

    ~cuda_host_to_device_assign_ck() { delete[] dst; }

    void single(char *dst, char *const *src)
    {
      ckernel_prefix *child = this->get_child_ckernel();
      expr_single_t single = child->get_function<expr_single_t>();

      single(this->dst, src, child);
      cuda_throw_if_not_success(
          cudaMemcpy(dst, this->dst, data_size, cudaMemcpyHostToDevice));
    }
  };

  struct cuda_host_to_device_copy_ck
      : nd::expr_ck<cuda_host_to_device_copy_ck, kernel_request_host, 1> {
    size_t data_size;

    cuda_host_to_device_copy_ck(size_t data_size) : data_size(data_size) {}

    void single(char *dst, char *const *src)
    {
      cuda_throw_if_not_success(
          cudaMemcpy(dst, *src, data_size, cudaMemcpyHostToDevice));
    }
  };

  struct cuda_device_to_host_assign_ck
      : nd::expr_ck<cuda_device_to_host_assign_ck, kernel_request_host, 1> {
    size_t data_size;
    char *src;

    cuda_device_to_host_assign_ck(size_t data_size)
        : data_size(data_size), src(new char[data_size])
    {
    }

    ~cuda_device_to_host_assign_ck() { delete[] src; }

    void single(char *dst, char *const *src)
    {
      ckernel_prefix *child = this->get_child_ckernel();
      expr_single_t single = child->get_function<expr_single_t>();

      cuda_throw_if_not_success(
          cudaMemcpy(this->src, *src, data_size, cudaMemcpyDeviceToHost));
      single(dst, &this->src, child);
    }
  };

  struct cuda_device_to_host_copy_ck
      : nd::expr_ck<cuda_device_to_host_copy_ck, kernel_request_host, 1> {
    size_t data_size;

    cuda_device_to_host_copy_ck(size_t data_size) : data_size(data_size) {}

    void single(char *dst, char *const *src)
    {
      cuda_throw_if_not_success(
          cudaMemcpy(dst, *src, data_size, cudaMemcpyDeviceToHost));
    }
  };

  struct cuda_device_to_device_copy_ck
      : nd::expr_ck<cuda_device_to_device_copy_ck, kernel_request_host, 1> {
    size_t data_size;

    cuda_device_to_device_copy_ck(size_t data_size) : data_size(data_size) {}

    void single(char *dst, char *const *src)
    {
      cuda_throw_if_not_success(
          cudaMemcpy(dst, *src, data_size, cudaMemcpyDeviceToDevice));
    }
  };

#endif

} // namespace dynd::nd
} // namespace dynd