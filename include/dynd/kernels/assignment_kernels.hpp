//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <stdexcept>

#include <dynd/fpstatus.hpp>
#include <dynd/type.hpp>
#include <dynd/typed_data_assign.hpp>
#include <dynd/func/assignment.hpp>
#include <dynd/kernels/cuda_launch.hpp>
#include <dynd/kernels/base_kernel.hpp>
#include <dynd/kernels/base_virtual_kernel.hpp>
#include <dynd/eval/eval_context.hpp>
#include <dynd/typed_data_assign.hpp>
#include <dynd/types/type_id.hpp>
#include <map>

#if defined(_MSC_VER)
// Tell the visual studio compiler we're accessing the FPU flags
#pragma fenv_access(on)
#endif

namespace dynd {

template <typename DstType, typename SrcType>
typename std::enable_if<(sizeof(DstType) < sizeof(SrcType)) &&
                            is_signed<DstType>::value &&
                            is_signed<SrcType>::value,
                        bool>::type
is_overflow(SrcType src)
{
  return src < static_cast<SrcType>(std::numeric_limits<DstType>::min()) ||
         src > static_cast<SrcType>(std::numeric_limits<DstType>::max());
}

template <typename DstType, typename SrcType>
typename std::enable_if<(sizeof(DstType) >= sizeof(SrcType)) &&
                            is_signed<DstType>::value &&
                            is_signed<SrcType>::value,
                        bool>::type
is_overflow(SrcType DYND_UNUSED(src))
{
  return false;
}

template <typename DstType, typename SrcType>
typename std::enable_if<(sizeof(DstType) < sizeof(SrcType)) &&
                            is_signed<DstType>::value &&
                            is_unsigned<SrcType>::value,
                        bool>::type
is_overflow(SrcType src)
{
  return src > static_cast<SrcType>(std::numeric_limits<DstType>::max());
}

template <typename DstType, typename SrcType>
typename std::enable_if<(sizeof(DstType) >= sizeof(SrcType)) &&
                            is_signed<DstType>::value &&
                            is_unsigned<SrcType>::value,
                        bool>::type
is_overflow(SrcType DYND_UNUSED(src))
{
  return false;
}

template <typename DstType, typename SrcType>
typename std::enable_if<(sizeof(DstType) < sizeof(SrcType)) &&
                            is_unsigned<DstType>::value &&
                            is_signed<SrcType>::value,
                        bool>::type
is_overflow(SrcType src)
{
  return src < static_cast<SrcType>(0) ||
         static_cast<SrcType>(std::numeric_limits<DstType>::max()) < src;
}

template <typename DstType, typename SrcType>
typename std::enable_if<(sizeof(DstType) >= sizeof(SrcType)) &&
                            is_unsigned<DstType>::value &&
                            is_signed<SrcType>::value,
                        bool>::type
is_overflow(SrcType src)
{
  return src < static_cast<SrcType>(0);
}

template <typename DstType, typename SrcType>
typename std::enable_if<(sizeof(DstType) < sizeof(SrcType)) &&
                            is_unsigned<DstType>::value &&
                            is_unsigned<SrcType>::value,
                        bool>::type
is_overflow(SrcType src)
{
  return static_cast<SrcType>(std::numeric_limits<DstType>::max()) < src;
}

template <typename DstType, typename SrcType>
typename std::enable_if<(sizeof(DstType) >= sizeof(SrcType)) &&
                            is_unsigned<DstType>::value &&
                            is_unsigned<SrcType>::value,
                        bool>::type
is_overflow(SrcType DYND_UNUSED(src))
{
  return false;
}

namespace nd {
  namespace detail {

    template <type_id_t DstTypeID, type_kind_t DstTypeKind,
              type_id_t Src0TypeID, type_kind_t Src0TypeKind,
              assign_error_mode... ErrorMode>
    struct assignment_kernel;

    template <type_id_t DstTypeID, type_kind_t DstTypeKind,
              type_id_t Src0TypeID, type_kind_t Src0TypeKind>
    struct assignment_kernel<DstTypeID, DstTypeKind, Src0TypeID, Src0TypeKind>
        : base_virtual_kernel<assignment_kernel<DstTypeID, DstTypeKind,
                                                Src0TypeID, Src0TypeKind>> {
      static intptr_t instantiate(
          const arrfunc_type_data *self, const ndt::arrfunc_type *self_tp,
          char *data, void *ckb, intptr_t ckb_offset, const ndt::type &dst_tp,
          const char *dst_arrmeta, intptr_t nsrc, const ndt::type *src_tp,
          const char *const *src_arrmeta, kernel_request_t kernreq,
          const eval::eval_context *ectx, const nd::array &kwds,
          const std::map<nd::string, ndt::type> &tp_vars)
      {
        switch (ectx->errmode) {
        case assign_error_nocheck:
          return assignment_kernel<
              DstTypeID, DstTypeKind, Src0TypeID, Src0TypeKind,
              assign_error_nocheck>::instantiate(self, self_tp, data, ckb,
                                                 ckb_offset, dst_tp,
                                                 dst_arrmeta, nsrc, src_tp,
                                                 src_arrmeta, kernreq, ectx,
                                                 kwds, tp_vars);
        case assign_error_overflow:
          return assignment_kernel<
              DstTypeID, DstTypeKind, Src0TypeID, Src0TypeKind,
              assign_error_overflow>::instantiate(self, self_tp, data, ckb,
                                                  ckb_offset, dst_tp,
                                                  dst_arrmeta, nsrc, src_tp,
                                                  src_arrmeta, kernreq, ectx,
                                                  kwds, tp_vars);
        case assign_error_fractional:
          return assignment_kernel<
              DstTypeID, DstTypeKind, Src0TypeID, Src0TypeKind,
              assign_error_fractional>::instantiate(self, self_tp, data, ckb,
                                                    ckb_offset, dst_tp,
                                                    dst_arrmeta, nsrc, src_tp,
                                                    src_arrmeta, kernreq, ectx,
                                                    kwds, tp_vars);
        case assign_error_inexact:
          return assignment_kernel<
              DstTypeID, DstTypeKind, Src0TypeID, Src0TypeKind,
              assign_error_inexact>::instantiate(self, self_tp, data, ckb,
                                                 ckb_offset, dst_tp,
                                                 dst_arrmeta, nsrc, src_tp,
                                                 src_arrmeta, kernreq, ectx,
                                                 kwds, tp_vars);
        default:
          throw std::runtime_error("error");
        }
      }

      static ndt::type make_type() {
        return ndt::type("(Any) -> Any");
      }
    };

    template <type_id_t DstTypeID, type_kind_t DstTypeKind,
              type_id_t Src0TypeID, type_kind_t Src0TypeKind,
              assign_error_mode ErrorMode>
    struct assignment_kernel<DstTypeID, DstTypeKind, Src0TypeID, Src0TypeKind,
                             ErrorMode>
        : base_kernel<assignment_kernel<DstTypeID, DstTypeKind, Src0TypeID,
                                        Src0TypeKind, ErrorMode>,
                      kernel_request_host, 1> {
      typedef typename type_of<DstTypeID>::type dst_type;
      typedef typename type_of<Src0TypeID>::type src_type;

      void single(char *dst, char *const *src)
      {
        DYND_TRACE_ASSIGNMENT(
            static_cast<dst_type>(*reinterpret_cast<src_type *>(src[0])),
            dst_type, *reinterpret_cast<src_type *>(src[0]), src_type);

        *reinterpret_cast<dst_type *>(dst) =
            static_cast<dst_type>(*reinterpret_cast<src_type *>(src[0]));
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

    // Complex floating point -> non-complex with no error checking
    template <type_id_t DstTypeID, type_id_t Src0TypeID>
    struct assignment_kernel<DstTypeID, sint_kind, Src0TypeID, complex_kind,
                             assign_error_nocheck>
        : base_kernel<assignment_kernel<DstTypeID, sint_kind, Src0TypeID,
                                        complex_kind, assign_error_nocheck>,
                      kernel_request_host, 1> {
      typedef typename type_of<DstTypeID>::type dst_type;
      typedef typename type_of<Src0TypeID>::type src0_type;

      void single(char *dst, char *const *src)
      {
        src0_type s = *reinterpret_cast<src0_type *>(src[0]);

        DYND_TRACE_ASSIGNMENT(static_cast<dst_type>(s.real()), dst_type, s,
                              src0_type);

        *reinterpret_cast<dst_type *>(dst) = static_cast<dst_type>(s.real());
      }
    };

    template <type_id_t DstTypeID, type_id_t Src0TypeID>
    struct assignment_kernel<DstTypeID, uint_kind, Src0TypeID, complex_kind,
                             assign_error_nocheck>
        : base_kernel<assignment_kernel<DstTypeID, uint_kind, Src0TypeID,
                                        complex_kind, assign_error_nocheck>,
                      kernel_request_host, 1> {
      typedef typename type_of<DstTypeID>::type dst_type;
      typedef typename type_of<Src0TypeID>::type src0_type;

      void single(char *dst, char *const *src)
      {
        src0_type s = *reinterpret_cast<src0_type *>(src[0]);

        DYND_TRACE_ASSIGNMENT(static_cast<dst_type>(s.real()), dst_type, s,
                              complex<src_real_type>);

        *reinterpret_cast<dst_type *>(dst) = static_cast<dst_type>(s.real());
      }
    };

    template <type_id_t DstTypeID, type_id_t Src0TypeID>
    struct assignment_kernel<DstTypeID, real_kind, Src0TypeID, complex_kind,
                             assign_error_nocheck>
        : base_kernel<assignment_kernel<DstTypeID, real_kind, Src0TypeID,
                                        complex_kind, assign_error_nocheck>,
                      kernel_request_host, 1> {
      typedef typename type_of<DstTypeID>::type dst_type;
      typedef typename type_of<Src0TypeID>::type src0_type;

      void single(char *dst, char *const *src)
      {
        src0_type s = *reinterpret_cast<src0_type *>(src[0]);

        DYND_TRACE_ASSIGNMENT(static_cast<dst_type>(s.real()), dst_type, s,
                              src0_type);

        *reinterpret_cast<dst_type *>(dst) = static_cast<dst_type>(s.real());
      }
    };

    // Signed int -> complex floating point with no checking
    template <type_id_t DstTypeID, type_id_t Src0TypeID>
    struct assignment_kernel<DstTypeID, complex_kind, Src0TypeID, sint_kind,
                             assign_error_nocheck>
        : base_kernel<assignment_kernel<DstTypeID, complex_kind, Src0TypeID,
                                        sint_kind, assign_error_nocheck>,
                      kernel_request_host, 1> {
      typedef typename type_of<DstTypeID>::type dst_type;
      typedef typename type_of<Src0TypeID>::type src0_type;

      void single(char *dst, char *const *src)
      {
        src0_type s = *reinterpret_cast<src0_type *>(src[0]);

        DYND_TRACE_ASSIGNMENT(d, dst_type, s, src0_type);

        *reinterpret_cast<dst_type *>(dst) =
            static_cast<typename dst_type::value_type>(s);
      }
    };

    // Signed int -> complex floating point with inexact checking
    template <type_id_t DstTypeID, type_id_t Src0TypeID>
    struct assignment_kernel<DstTypeID, complex_kind, Src0TypeID, sint_kind,
                             assign_error_inexact>
        : base_kernel<assignment_kernel<DstTypeID, complex_kind, Src0TypeID,
                                        sint_kind, assign_error_inexact>,
                      kernel_request_host, 1> {
      typedef typename type_of<DstTypeID>::type dst_type;
      typedef typename type_of<Src0TypeID>::type src0_type;

      void single(char *dst, char *const *src)
      {
        src0_type s = *reinterpret_cast<src0_type *>(src[0]);
        typename dst_type::value_type d =
            static_cast<typename dst_type::value_type>(s);

        DYND_TRACE_ASSIGNMENT(d, dst_type, s, src0_type);

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

    // Signed int -> complex floating point with other checking
    template <type_id_t DstTypeID, type_id_t Src0TypeID>
    struct assignment_kernel<DstTypeID, complex_kind, Src0TypeID, sint_kind,
                             assign_error_overflow>
        : assignment_kernel<DstTypeID, complex_kind, Src0TypeID, sint_kind,
                            assign_error_nocheck> {
    };

    template <type_id_t DstTypeID, type_id_t Src0TypeID>
    struct assignment_kernel<DstTypeID, complex_kind, Src0TypeID, sint_kind,
                             assign_error_fractional>
        : assignment_kernel<DstTypeID, complex_kind, Src0TypeID, sint_kind,
                            assign_error_nocheck> {
    };

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

    // Unsigned int -> floating point with inexact checking
    template <type_id_t DstTypeID, type_id_t Src0TypeID>
    struct assignment_kernel<DstTypeID, real_kind, Src0TypeID, uint_kind,
                             assign_error_inexact>
        : base_kernel<assignment_kernel<DstTypeID, real_kind, Src0TypeID,
                                        uint_kind, assign_error_inexact>,
                      kernel_request_host, 1> {
      typedef typename type_of<DstTypeID>::type dst_type;
      typedef typename type_of<Src0TypeID>::type src0_type;

      void single(char *dst, char *const *src)
      {
        src0_type s = *reinterpret_cast<src0_type *>(src[0]);
        dst_type d = static_cast<dst_type>(s);

        DYND_TRACE_ASSIGNMENT(d, dst_type, s, src0_type);

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

    // Unsigned int -> floating point with other checking
    template <type_id_t DstTypeID, type_id_t Src0TypeID>
    struct assignment_kernel<DstTypeID, real_kind, Src0TypeID, uint_kind,
                             assign_error_overflow>
        : assignment_kernel<DstTypeID, real_kind, Src0TypeID, uint_kind,
                            assign_error_nocheck> {
    };

    template <type_id_t DstTypeID, type_id_t Src0TypeID>
    struct assignment_kernel<DstTypeID, real_kind, Src0TypeID, uint_kind,
                             assign_error_fractional>
        : assignment_kernel<DstTypeID, real_kind, Src0TypeID, uint_kind,
                            assign_error_nocheck> {
    };

    // Unsigned int -> complex floating point with no checking
    template <type_id_t DstTypeID, type_id_t Src0TypeID>
    struct assignment_kernel<DstTypeID, complex_kind, Src0TypeID, uint_kind,
                             assign_error_nocheck>
        : base_kernel<assignment_kernel<DstTypeID, complex_kind, Src0TypeID,
                                        uint_kind, assign_error_nocheck>,
                      kernel_request_host, 1> {
      typedef typename type_of<DstTypeID>::type dst_type;
      typedef typename type_of<Src0TypeID>::type src0_type;

      DYND_CUDA_HOST_DEVICE void single(char *dst, char *const *src)
      {
        src0_type s = *reinterpret_cast<src0_type *>(src[0]);

        DYND_TRACE_ASSIGNMENT(d, dst_type, s, src0_type);

        *reinterpret_cast<dst_type *>(dst) =
            static_cast<typename dst_type::value_type>(s);
      }
    };

    // Unsigned int -> complex floating point with inexact checking
    template <type_id_t DstTypeID, type_id_t Src0TypeID>
    struct assignment_kernel<DstTypeID, complex_kind, Src0TypeID, uint_kind,
                             assign_error_inexact>
        : base_kernel<assignment_kernel<DstTypeID, complex_kind, Src0TypeID,
                                        uint_kind, assign_error_inexact>,
                      kernel_request_host, 1> {
      typedef typename type_of<DstTypeID>::type dst_type;
      typedef typename type_of<Src0TypeID>::type src0_type;

      void single(char *dst, char *const *src)
      {
        src0_type s = *reinterpret_cast<src0_type *>(src[0]);
        typename dst_type::value_type d =
            static_cast<typename dst_type::value_type>(s);

        DYND_TRACE_ASSIGNMENT(d, dst_type, s, src0_type);

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

    // Unsigned int -> complex floating point with other checking
    template <type_id_t DstTypeID, type_id_t Src0TypeID>
    struct assignment_kernel<DstTypeID, complex_kind, Src0TypeID, uint_kind,
                             assign_error_overflow>
        : assignment_kernel<DstTypeID, complex_kind, Src0TypeID, uint_kind,
                            assign_error_nocheck> {
    };

    template <type_id_t DstTypeID, type_id_t Src0TypeID>
    struct assignment_kernel<DstTypeID, complex_kind, Src0TypeID, uint_kind,
                             assign_error_fractional>
        : assignment_kernel<DstTypeID, complex_kind, Src0TypeID, uint_kind,
                            assign_error_nocheck> {
    };

    // Floating point -> signed int with overflow checking
    template <type_id_t DstTypeID, type_id_t Src0TypeID>
    struct assignment_kernel<DstTypeID, sint_kind, Src0TypeID, real_kind,
                             assign_error_overflow>
        : base_kernel<assignment_kernel<DstTypeID, sint_kind, Src0TypeID,
                                        real_kind, assign_error_overflow>,
                      kernel_request_host, 1> {
      typedef typename type_of<DstTypeID>::type dst_type;
      typedef typename type_of<Src0TypeID>::type src0_type;

      void single(char *dst, char *const *src)
      {
        src0_type s = *reinterpret_cast<src0_type *>(src[0]);

        DYND_TRACE_ASSIGNMENT(static_cast<dst_type>(s), dst_type, s, src_type);

        if (s < std::numeric_limits<dst_type>::min() ||
            std::numeric_limits<dst_type>::max() < s) {
          std::stringstream ss;
          ss << "overflow while assigning " << ndt::make_type<src0_type>()
             << " value ";
          ss << s << " to " << ndt::make_type<dst_type>();
          throw std::overflow_error(ss.str());
        }
        *reinterpret_cast<dst_type *>(dst) = static_cast<dst_type>(s);
      }
    };

    // Floating point -> signed int with fractional checking
    template <type_id_t DstTypeID, type_id_t Src0TypeID>
    struct assignment_kernel<DstTypeID, sint_kind, Src0TypeID, real_kind,
                             assign_error_fractional>
        : base_kernel<assignment_kernel<DstTypeID, sint_kind, Src0TypeID,
                                        real_kind, assign_error_fractional>,
                      kernel_request_host, 1> {
      typedef typename type_of<DstTypeID>::type dst_type;
      typedef typename type_of<Src0TypeID>::type src0_type;

      void single(char *dst, char *const *src)
      {
        src0_type s = *reinterpret_cast<src0_type *>(src[0]);

        DYND_TRACE_ASSIGNMENT(static_cast<dst_type>(s), dst_type, s, src0_type);

        if (s < std::numeric_limits<dst_type>::min() ||
            std::numeric_limits<dst_type>::max() < s) {
          std::stringstream ss;
          ss << "overflow while assigning " << ndt::make_type<src0_type>()
             << " value ";
          ss << s << " to " << ndt::make_type<dst_type>();
          throw std::overflow_error(ss.str());
        }

        if (floor(s) != s) {
          std::stringstream ss;
          ss << "fractional part lost while assigning "
             << ndt::make_type<src0_type>() << " value ";
          ss << s << " to " << ndt::make_type<dst_type>();
          throw std::runtime_error(ss.str());
        }
        *reinterpret_cast<dst_type *>(dst) = static_cast<dst_type>(s);
      }
    };

    // Floating point -> signed int with other checking
    template <type_id_t DstTypeID, type_id_t Src0TypeID>
    struct assignment_kernel<DstTypeID, sint_kind, Src0TypeID, real_kind,
                             assign_error_inexact>
        : assignment_kernel<DstTypeID, sint_kind, Src0TypeID, real_kind,
                            assign_error_fractional> {
    };

    // Complex floating point -> signed int with overflow checking
    template <type_id_t DstTypeID, type_id_t Src0TypeID>
    struct assignment_kernel<DstTypeID, sint_kind, Src0TypeID, complex_kind,
                             assign_error_overflow>
        : base_kernel<assignment_kernel<DstTypeID, sint_kind, Src0TypeID,
                                        complex_kind, assign_error_overflow>,
                      kernel_request_host, 1> {
      typedef typename type_of<DstTypeID>::type dst_type;
      typedef typename type_of<Src0TypeID>::type src0_type;

      void single(char *dst, char *const *src)
      {
        src0_type s = *reinterpret_cast<src0_type *>(src[0]);

        DYND_TRACE_ASSIGNMENT(static_cast<dst_type>(s.real()), dst_type, s,
                              src0_type);

        if (s.imag() != 0) {
          std::stringstream ss;
          ss << "loss of imaginary component while assigning "
             << ndt::make_type<src0_type>() << " value ";
          ss << s << " to " << ndt::make_type<dst_type>();
          throw std::runtime_error(ss.str());
        }

        if (s.real() < std::numeric_limits<dst_type>::min() ||
            std::numeric_limits<dst_type>::max() < s.real()) {
          std::stringstream ss;
          ss << "overflow while assigning " << ndt::make_type<src0_type>()
             << " value ";
          ss << s << " to " << ndt::make_type<dst_type>();
          throw std::overflow_error(ss.str());
        }
        *reinterpret_cast<dst_type *>(dst) = static_cast<dst_type>(s.real());
      }
    };

    // Complex floating point -> signed int with fractional checking
    template <type_id_t DstTypeID, type_id_t Src0TypeID>
    struct assignment_kernel<DstTypeID, sint_kind, Src0TypeID, complex_kind,
                             assign_error_fractional>
        : base_kernel<assignment_kernel<DstTypeID, sint_kind, Src0TypeID,
                                        complex_kind, assign_error_fractional>,
                      kernel_request_host, 1> {
      typedef typename type_of<DstTypeID>::type dst_type;
      typedef typename type_of<Src0TypeID>::type src0_type;

      void single(char *dst, char *const *src)
      {
        src0_type s = *reinterpret_cast<src0_type *>(src[0]);

        DYND_TRACE_ASSIGNMENT(static_cast<dst_type>(s.real()), dst_type, s,
                              src0_type);

        if (s.imag() != 0) {
          std::stringstream ss;
          ss << "loss of imaginary component while assigning "
             << ndt::make_type<src0_type>() << " value ";
          ss << s << " to " << ndt::make_type<dst_type>();
          throw std::runtime_error(ss.str());
        }

        if (s.real() < std::numeric_limits<dst_type>::min() ||
            std::numeric_limits<dst_type>::max() < s.real()) {
          std::stringstream ss;
          ss << "overflow while assigning " << ndt::make_type<src0_type>()
             << " value ";
          ss << s << " to " << ndt::make_type<dst_type>();
          throw std::overflow_error(ss.str());
        }

        if (std::floor(s.real()) != s.real()) {
          std::stringstream ss;
          ss << "fractional part lost while assigning "
             << ndt::make_type<src0_type>() << " value ";
          ss << s << " to " << ndt::make_type<dst_type>();
          throw std::runtime_error(ss.str());
        }
        *reinterpret_cast<dst_type *>(dst) = static_cast<dst_type>(s.real());
      }
    };

    // Complex floating point -> signed int with other checking
    template <type_id_t DstTypeID, type_id_t Src0TypeID>
    struct assignment_kernel<DstTypeID, sint_kind, Src0TypeID, complex_kind,
                             assign_error_inexact>
        : assignment_kernel<DstTypeID, sint_kind, Src0TypeID, complex_kind,
                            assign_error_fractional> {
    };

    // Floating point -> unsigned int with overflow checking
    template <type_id_t DstTypeID, type_id_t Src0TypeID>
    struct assignment_kernel<DstTypeID, uint_kind, Src0TypeID, real_kind,
                             assign_error_overflow>
        : base_kernel<assignment_kernel<DstTypeID, uint_kind, Src0TypeID,
                                        real_kind, assign_error_overflow>,
                      kernel_request_host, 1> {
      typedef typename type_of<DstTypeID>::type dst_type;
      typedef typename type_of<Src0TypeID>::type src0_type;

      void single(char *dst, char *const *src)
      {
        src0_type s = *reinterpret_cast<src0_type *>(src[0]);

        DYND_TRACE_ASSIGNMENT(static_cast<dst_type>(s), dst_type, s, src0_type);

        if (s < 0 || std::numeric_limits<dst_type>::max() < s) {
          std::stringstream ss;
          ss << "overflow while assigning " << ndt::make_type<src0_type>()
             << " value ";
          ss << s << " to " << ndt::make_type<dst_type>();
          throw std::overflow_error(ss.str());
        }
        *reinterpret_cast<dst_type *>(dst) = static_cast<dst_type>(s);
      }
    };

    // Floating point -> unsigned int with fractional checking
    template <type_id_t DstTypeID, type_id_t Src0TypeID>
    struct assignment_kernel<DstTypeID, uint_kind, Src0TypeID, real_kind,
                             assign_error_fractional>
        : base_kernel<assignment_kernel<DstTypeID, uint_kind, Src0TypeID,
                                        real_kind, assign_error_fractional>,
                      kernel_request_host, 1> {
      typedef typename type_of<DstTypeID>::type dst_type;
      typedef typename type_of<Src0TypeID>::type src0_type;

      void single(char *dst, char *const *src)
      {
        src0_type s = *reinterpret_cast<src0_type *>(src[0]);

        DYND_TRACE_ASSIGNMENT(static_cast<dst_type>(s), dst_type, s, src0_type);

        if (s < 0 || std::numeric_limits<dst_type>::max() < s) {
          std::stringstream ss;
          ss << "overflow while assigning " << ndt::make_type<src0_type>()
             << " value ";
          ss << s << " to " << ndt::make_type<dst_type>();
          throw std::overflow_error(ss.str());
        }

        if (floor(s) != s) {
          std::stringstream ss;
          ss << "fractional part lost while assigning "
             << ndt::make_type<src0_type>() << " value ";
          ss << s << " to " << ndt::make_type<dst_type>();
          throw std::runtime_error(ss.str());
        }
        *reinterpret_cast<dst_type *>(dst) = static_cast<dst_type>(s);
      }
    };

    // Floating point -> unsigned int with other checking
    template <type_id_t DstTypeID, type_id_t Src0TypeID>
    struct assignment_kernel<DstTypeID, uint_kind, Src0TypeID, real_kind,
                             assign_error_inexact>
        : assignment_kernel<DstTypeID, uint_kind, Src0TypeID, real_kind,
                            assign_error_fractional> {
    };

    // Complex floating point -> unsigned int with overflow checking
    template <type_id_t DstTypeID, type_id_t Src0TypeID>
    struct assignment_kernel<DstTypeID, uint_kind, Src0TypeID, complex_kind,
                             assign_error_overflow>
        : base_kernel<assignment_kernel<DstTypeID, uint_kind, Src0TypeID,
                                        complex_kind, assign_error_overflow>,
                      kernel_request_host, 1> {
      typedef typename type_of<DstTypeID>::type dst_type;
      typedef typename type_of<Src0TypeID>::type src0_type;

      void single(char *dst, char *const *src)
      {
        src0_type s = *reinterpret_cast<src0_type *>(src[0]);

        DYND_TRACE_ASSIGNMENT(static_cast<dst_type>(s.real()), dst_type, s,
                              complex<src_real_type>);

        if (s.imag() != 0) {
          std::stringstream ss;
          ss << "loss of imaginary component while assigning "
             << ndt::make_type<src0_type>() << " value ";
          ss << s << " to " << ndt::make_type<dst_type>();
          throw std::runtime_error(ss.str());
        }

        if (s.real() < 0 || std::numeric_limits<dst_type>::max() < s.real()) {
          std::stringstream ss;
          ss << "overflow while assigning " << ndt::make_type<src0_type>()
             << " value ";
          ss << s << " to " << ndt::make_type<dst_type>();
          throw std::overflow_error(ss.str());
        }
        *reinterpret_cast<dst_type *>(dst) = static_cast<dst_type>(s.real());
      }
    };

    // Complex floating point -> unsigned int with fractional checking
    template <type_id_t DstTypeID, type_id_t Src0TypeID>
    struct assignment_kernel<DstTypeID, uint_kind, Src0TypeID, complex_kind,
                             assign_error_fractional>
        : base_kernel<assignment_kernel<DstTypeID, uint_kind, Src0TypeID,
                                        complex_kind, assign_error_fractional>,
                      kernel_request_host, 1> {
      typedef typename type_of<DstTypeID>::type dst_type;
      typedef typename type_of<Src0TypeID>::type src0_type;

      void single(char *dst, char *const *src)
      {
        src0_type s = *reinterpret_cast<src0_type *>(src[0]);

        DYND_TRACE_ASSIGNMENT(static_cast<dst_type>(s.real()), dst_type, s,
                              src0_type);

        if (s.imag() != 0) {
          std::stringstream ss;
          ss << "loss of imaginary component while assigning "
             << ndt::make_type<src0_type>() << " value ";
          ss << s << " to " << ndt::make_type<dst_type>();
          throw std::runtime_error(ss.str());
        }

        if (s.real() < 0 || std::numeric_limits<dst_type>::max() < s.real()) {
          std::stringstream ss;
          ss << "overflow while assigning " << ndt::make_type<src0_type>()
             << " value ";
          ss << s << " to " << ndt::make_type<dst_type>();
          throw std::overflow_error(ss.str());
        }

        if (std::floor(s.real()) != s.real()) {
          std::stringstream ss;
          ss << "fractional part lost while assigning "
             << ndt::make_type<src0_type>() << " value ";
          ss << s << " to " << ndt::make_type<dst_type>();
          throw std::runtime_error(ss.str());
        }
        *reinterpret_cast<dst_type *>(dst) = static_cast<dst_type>(s.real());
      }
    };

    // Complex floating point -> unsigned int with other checking
    template <type_id_t DstTypeID, type_id_t Src0TypeID>
    struct assignment_kernel<DstTypeID, uint_kind, Src0TypeID, complex_kind,
                             assign_error_inexact>
        : assignment_kernel<DstTypeID, uint_kind, Src0TypeID, complex_kind,
                            assign_error_fractional> {
    };

    // float -> float with no checking
    template <>
    struct assignment_kernel<float32_type_id, real_kind, float32_type_id,
                             real_kind, assign_error_overflow>
        : assignment_kernel<float32_type_id, real_kind, float32_type_id,
                            real_kind, assign_error_nocheck> {
    };

    template <>
    struct assignment_kernel<float32_type_id, real_kind, float32_type_id,
                             real_kind, assign_error_fractional>
        : assignment_kernel<float32_type_id, real_kind, float32_type_id,
                            real_kind, assign_error_nocheck> {
    };

    template <>
    struct assignment_kernel<float32_type_id, real_kind, float32_type_id,
                             real_kind, assign_error_inexact>
        : assignment_kernel<float32_type_id, real_kind, float32_type_id,
                            real_kind, assign_error_nocheck> {
    };

    // complex<float> -> complex<float> with no checking
    template <>
    struct assignment_kernel<complex_float32_type_id, complex_kind,
                             complex_float32_type_id, complex_kind,
                             assign_error_overflow>
        : assignment_kernel<complex_float32_type_id, complex_kind,
                            complex_float32_type_id, complex_kind,
                            assign_error_nocheck> {
    };

    template <>
    struct assignment_kernel<complex_float32_type_id, complex_kind,
                             complex_float32_type_id, complex_kind,
                             assign_error_fractional>
        : assignment_kernel<complex_float32_type_id, complex_kind,
                            complex_float32_type_id, complex_kind,
                            assign_error_nocheck> {
    };

    template <>
    struct assignment_kernel<complex_float32_type_id, complex_kind,
                             complex_float32_type_id, complex_kind,
                             assign_error_inexact>
        : assignment_kernel<complex_float32_type_id, complex_kind,
                            complex_float32_type_id, complex_kind,
                            assign_error_nocheck> {
    };

    // float -> double with no checking
    template <>
    struct assignment_kernel<float64_type_id, real_kind, float32_type_id,
                             real_kind, assign_error_overflow>
        : assignment_kernel<float64_type_id, real_kind, float32_type_id,
                            real_kind, assign_error_nocheck> {
    };

    template <>
    struct assignment_kernel<float64_type_id, real_kind, float32_type_id,
                             real_kind, assign_error_fractional>
        : assignment_kernel<float64_type_id, real_kind, float32_type_id,
                            real_kind, assign_error_nocheck> {
    };

    template <>
    struct assignment_kernel<float64_type_id, real_kind, float32_type_id,
                             real_kind, assign_error_inexact>
        : assignment_kernel<float64_type_id, real_kind, float32_type_id,
                            real_kind, assign_error_nocheck> {
    };

    // complex<float> -> complex<double> with no checking
    template <>
    struct assignment_kernel<complex_float64_type_id, complex_kind,
                             complex_float32_type_id, complex_kind,
                             assign_error_overflow>
        : assignment_kernel<complex_float64_type_id, complex_kind,
                            complex_float32_type_id, complex_kind,
                            assign_error_nocheck> {
    };

    template <>
    struct assignment_kernel<complex_float64_type_id, complex_kind,
                             complex_float32_type_id, complex_kind,
                             assign_error_fractional>
        : assignment_kernel<complex_float64_type_id, complex_kind,
                            complex_float32_type_id, complex_kind,
                            assign_error_nocheck> {
    };

    template <>
    struct assignment_kernel<complex_float64_type_id, complex_kind,
                             complex_float32_type_id, complex_kind,
                             assign_error_inexact>
        : assignment_kernel<complex_float64_type_id, complex_kind,
                            complex_float32_type_id, complex_kind,
                            assign_error_nocheck> {
    };

    // double -> double with no checking
    template <>
    struct assignment_kernel<float64_type_id, real_kind, float64_type_id,
                             real_kind, assign_error_overflow>
        : assignment_kernel<float64_type_id, real_kind, float64_type_id,
                            real_kind, assign_error_nocheck> {
    };

    template <>
    struct assignment_kernel<float64_type_id, real_kind, float64_type_id,
                             real_kind, assign_error_fractional>
        : assignment_kernel<float64_type_id, real_kind, float64_type_id,
                            real_kind, assign_error_nocheck> {
    };

    template <>
    struct assignment_kernel<float64_type_id, real_kind, float64_type_id,
                             real_kind, assign_error_inexact>
        : assignment_kernel<float64_type_id, real_kind, float64_type_id,
                            real_kind, assign_error_nocheck> {
    };

    // complex<double> -> complex<double> with no checking
    template <>
    struct assignment_kernel<complex_float64_type_id, complex_kind,
                             complex_float64_type_id, complex_kind,
                             assign_error_overflow>
        : assignment_kernel<complex_float64_type_id, complex_kind,
                            complex_float64_type_id, complex_kind,
                            assign_error_nocheck> {
    };

    template <>
    struct assignment_kernel<complex_float64_type_id, complex_kind,
                             complex_float64_type_id, complex_kind,
                             assign_error_fractional>
        : assignment_kernel<complex_float64_type_id, complex_kind,
                            complex_float64_type_id, complex_kind,
                            assign_error_nocheck> {
    };

    template <>
    struct assignment_kernel<complex_float64_type_id, complex_kind,
                             complex_float64_type_id, complex_kind,
                             assign_error_inexact>
        : assignment_kernel<complex_float64_type_id, complex_kind,
                            complex_float64_type_id, complex_kind,
                            assign_error_nocheck> {
    };

    /*
      // double -> float with overflow checking
      template <type_id_t DstTypeID, type_id_t Src0TypeID>
      struct assignment_kernel<DstTypeID, real_kind, Src0TypeID,
                               real_kind, assign_error_overflow>
          : base_kernel<
                assignment_kernel<DstTypeID, real_kind, Src0TypeID,
                                  real_kind, assign_error_overflow>,
                kernel_request_host, 1> {
        typedef typename type_of<DstTypeID>::type dst_type;
        typedef typename type_of<Src0TypeID>::type src0_type;

        void single(char *dst, char *const *src)
        {
          src0_type s = *reinterpret_cast<src0_type *>(src[0]);

          DYND_TRACE_ASSIGNMENT(static_cast<dst_type>(s), dst_type, s,
    src0_type);

    #if defined(DYND_USE_FPSTATUS)
          clear_fp_status();
          *reinterpret_cast<dst_type *>(dst) = static_cast<dst_type>(s);
          if (is_overflow_fp_status()) {
            std::stringstream ss;
            ss << "overflow while assigning " << ndt::make_type<src0_type>()
               << " value ";
            ss << s << " to " << ndt::make_type<dst_type>();
            throw std::overflow_error(ss.str());
          }
    #else
          src0_type sd = s;
          if (isfinite(sd) && (sd < -std::numeric_limits<dst_type>::max() ||
                               sd > std::numeric_limits<dst_type>::max())) {
            std::stringstream ss;
            ss << "overflow while assigning " << ndt::make_type<src0_type>()
               << " value ";
            ss << s << " to " << ndt::make_type<dst_type>();
            throw std::overflow_error(ss.str());
          }
          *reinterpret_cast<dst_type *>(dst) = static_cast<dst_type>(sd);
    #endif // DYND_USE_FPSTATUS
        }
      };
    */

    // real -> real with overflow checking
    template <type_id_t DstTypeID, type_id_t Src0TypeID>
    struct assignment_kernel<DstTypeID, real_kind, Src0TypeID, real_kind,
                             assign_error_overflow>
        : base_kernel<assignment_kernel<DstTypeID, real_kind, Src0TypeID,
                                        real_kind, assign_error_overflow>,
                      kernel_request_host, 1> {
      typedef typename type_of<DstTypeID>::type dst_type;
      typedef typename type_of<Src0TypeID>::type src0_type;

      void single(char *dst, char *const *src)
      {
        src0_type s = *reinterpret_cast<src0_type *>(src[0]);

        DYND_TRACE_ASSIGNMENT(static_cast<dst_type>(s), dst_type, s, src0_type);

#if defined(DYND_USE_FPSTATUS)
        clear_fp_status();
        *reinterpret_cast<dst_type *>(dst) = static_cast<dst_type>(s);
        if (is_overflow_fp_status()) {
          std::stringstream ss;
          ss << "overflow while assigning " << ndt::make_type<src0_type>()
             << " value ";
          ss << s << " to " << ndt::make_type<dst_type>();
          throw std::overflow_error(ss.str());
        }
#else
        src0_type sd = s;
        if (isfinite(sd) && (sd < -std::numeric_limits<dst_type>::max() ||
                             sd > std::numeric_limits<dst_type>::max())) {
          std::stringstream ss;
          ss << "overflow while assigning " << ndt::make_type<src0_type>()
             << " value ";
          ss << s << " to " << ndt::make_type<dst_type>();
          throw std::overflow_error(ss.str());
        }
        *reinterpret_cast<dst_type *>(dst) = static_cast<dst_type>(sd);
#endif // DYND_USE_FPSTATUS
      }
    };

    // real -> real with fractional checking
    template <type_id_t DstTypeID, type_id_t Src0TypeID>
    struct assignment_kernel<DstTypeID, real_kind, Src0TypeID, real_kind,
                             assign_error_fractional>
        : assignment_kernel<DstTypeID, real_kind, Src0TypeID, real_kind,
                            assign_error_overflow> {
    };

    // real -> real with inexact checking
    template <type_id_t DstTypeID, type_id_t Src0TypeID>
    struct assignment_kernel<DstTypeID, real_kind, Src0TypeID, real_kind,
                             assign_error_inexact>
        : base_kernel<assignment_kernel<DstTypeID, real_kind, Src0TypeID,
                                        real_kind, assign_error_inexact>,
                      kernel_request_host, 1> {
      typedef typename type_of<DstTypeID>::type dst_type;
      typedef typename type_of<Src0TypeID>::type src0_type;

      void single(char *dst, char *const *src)
      {
        src0_type s = *reinterpret_cast<src0_type *>(src[0]);

        DYND_TRACE_ASSIGNMENT(static_cast<dst_type>(s), dst_type, s, src0_type);

        dst_type d;
#if defined(DYND_USE_FPSTATUS)
        clear_fp_status();
        d = static_cast<dst_type>(s);
        if (is_overflow_fp_status()) {
          std::stringstream ss;
          ss << "overflow while assigning " << ndt::make_type<src0_type>()
             << " value ";
          ss << s << " to " << ndt::make_type<dst_type>();
          throw std::overflow_error(ss.str());
        }
#else
        if (isfinite(s) && (s < -std::numeric_limits<dst_type>::max() ||
                            s > std::numeric_limits<dst_type>::max())) {
          std::stringstream ss;
          ss << "overflow while assigning " << ndt::make_type<src0_type>()
             << " value ";
          ss << s << " to " << ndt::make_type<dst_type>();
          throw std::runtime_error(ss.str());
        }
        d = static_cast<dst_type>(s);
#endif // DYND_USE_FPSTATUS

        // The inexact status didn't work as it should have, so converting back
        // to
        // double and comparing
        // if (is_inexact_fp_status()) {
        //    throw std::runtime_error("inexact precision loss while assigning
        //    double to float");
        //}
        if (d != s) {
          std::stringstream ss;
          ss << "inexact precision loss while assigning "
             << ndt::make_type<src0_type>() << " value ";
          ss << s << " to " << ndt::make_type<dst_type>();
          throw std::runtime_error(ss.str());
        }
        *reinterpret_cast<dst_type *>(dst) = d;
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

    // Unsigned int -> signed int with overflow checking just when sizeof(dst)
    // <=
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

    // Unsigned int -> unsigned int with overflow checking just when sizeof(dst)
    // <
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

        DYND_TRACE_ASSIGNMENT(d, dst_type, s, src0_type);

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

    template <type_id_t DstTypeID, type_id_t Src0TypeID,
              assign_error_mode ErrorMode>
    struct assignment_kernel<DstTypeID, complex_kind, Src0TypeID, real_kind,
                             ErrorMode>
        : base_kernel<assignment_kernel<DstTypeID, complex_kind, Src0TypeID,
                                        real_kind, ErrorMode>,
                      kernel_request_host, 1> {
      typedef typename type_of<DstTypeID>::type dst_type;
      typedef typename type_of<Src0TypeID>::type src0_type;

      void single(char *dst, char *const *src)
      {
        src0_type s = *reinterpret_cast<src0_type *>(src[0]);

        DYND_TRACE_ASSIGNMENT(static_cast<dst_type>(s), dst_type, s, src0_type);

        *reinterpret_cast<dst_type *>(dst) =
            static_cast<typename dst_type::value_type>(s);
      }
    };

    // complex -> real with overflow checking
    template <type_id_t DstTypeID, type_id_t Src0TypeID>
    struct assignment_kernel<DstTypeID, real_kind, Src0TypeID, complex_kind,
                             assign_error_overflow>
        : base_kernel<assignment_kernel<DstTypeID, real_kind, Src0TypeID,
                                        complex_kind, assign_error_overflow>,
                      kernel_request_host, 1> {
      typedef typename type_of<DstTypeID>::type dst_type;
      typedef typename type_of<Src0TypeID>::type src0_type;

      void single(char *dst, char *const *src)
      {
        src0_type s = *reinterpret_cast<src0_type *>(src[0]);
        dst_type d;

        DYND_TRACE_ASSIGNMENT(static_cast<dst_type>(s.real()), dst_type, s,
                              src0_type);

        if (s.imag() != 0) {
          std::stringstream ss;
          ss << "loss of imaginary component while assigning "
             << ndt::make_type<src0_type>() << " value ";
          ss << *src << " to " << ndt::make_type<dst_type>();
          throw std::runtime_error(ss.str());
        }

#if defined(DYND_USE_FPSTATUS)
        clear_fp_status();
        d = static_cast<dst_type>(s.real());
        if (is_overflow_fp_status()) {
          std::stringstream ss;
          ss << "overflow while assigning " << ndt::make_type<src0_type>()
             << " value ";
          ss << *src << " to " << ndt::make_type<dst_type>();
          throw std::overflow_error(ss.str());
        }
#else
        if (s.real() < -std::numeric_limits<dst_type>::max() ||
            s.real() > std::numeric_limits<dst_type>::max()) {
          std::stringstream ss;
          ss << "overflow while assigning " << ndt::make_type<src0_type>()
             << " value ";
          ss << *src << " to " << ndt::make_type<dst_type>();
          throw std::overflow_error(ss.str());
        }
        d = static_cast<dst_type>(s.real());
#endif // DYND_USE_FPSTATUS

        *reinterpret_cast<dst_type *>(dst) = d;
      }
    };

    // complex -> real with inexact checking
    template <type_id_t DstTypeID, type_id_t Src0TypeID>
    struct assignment_kernel<DstTypeID, real_kind, Src0TypeID, complex_kind,
                             assign_error_inexact>
        : assignment_kernel<DstTypeID, real_kind, Src0TypeID, complex_kind,
                            assign_error_overflow> {
    };

    // complex -> real with fractional checking
    template <type_id_t DstTypeID, type_id_t SrcTypeID>
    struct assignment_kernel<DstTypeID, real_kind, SrcTypeID, complex_kind,
                             assign_error_fractional>
        : assignment_kernel<DstTypeID, real_kind, SrcTypeID, complex_kind,
                            assign_error_overflow> {
    };

    // complex<double> -> float with inexact checking
    template <>
    struct assignment_kernel<float32_type_id, real_kind,
                             complex_float64_type_id, complex_kind,
                             assign_error_inexact>
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

    // real -> complex with overflow checking
    template <type_id_t DstTypeID, type_id_t Src0TypeID>
    struct assignment_kernel<DstTypeID, complex_kind, Src0TypeID, real_kind,
                             assign_error_overflow>
        : base_kernel<assignment_kernel<DstTypeID, complex_kind, Src0TypeID,
                                        real_kind, assign_error_overflow>,
                      kernel_request_host, 1> {
      typedef typename type_of<DstTypeID>::type dst_type;
      typedef typename type_of<Src0TypeID>::type src0_type;

      void single(char *dst, char *const *src)
      {
        src0_type s = *reinterpret_cast<src0_type *>(src[0]);
        typename dst_type::value_type d;

        DYND_TRACE_ASSIGNMENT(static_cast<dst_type>(s), dst_type, s, src0_type);

#if defined(DYND_USE_FPSTATUS)
        clear_fp_status();
        d = static_cast<typename dst_type::value_type>(s);
        if (is_overflow_fp_status()) {
          std::stringstream ss;
          ss << "overflow while assigning " << ndt::make_type<src0_type>()
             << " value ";
          ss << s << " to " << ndt::make_type<dst_type>();
          throw std::overflow_error(ss.str());
        }
#else
        if (isfinite(s) &&
            (s < -std::numeric_limits<typename dst_type::value_type>::max() ||
             s > std::numeric_limits<typename dst_type::value_type>::max())) {
          std::stringstream ss;
          ss << "overflow while assigning " << ndt::make_type<src0_type>()
             << " value ";
          ss << s << " to " << ndt::make_type<dst_type>();
          throw std::overflow_error(ss.str());
        }
        d = static_cast<typename dst_type::value_type>(s);
#endif // DYND_USE_FPSTATUS

        *reinterpret_cast<dst_type *>(dst) = d;
      }
    };

    // real -> complex with fractional checking
    template <type_id_t DstTypeID, type_id_t Src0TypeID>
    struct assignment_kernel<DstTypeID, complex_kind, Src0TypeID, real_kind,
                             assign_error_fractional>
        : assignment_kernel<DstTypeID, complex_kind, Src0TypeID, real_kind,
                            assign_error_overflow> {
    };

    // real -> complex with inexact checking
    template <type_id_t DstTypeID, type_id_t Src0TypeID>
    struct assignment_kernel<DstTypeID, complex_kind, Src0TypeID, real_kind,
                             assign_error_inexact>
        : base_kernel<assignment_kernel<DstTypeID, complex_kind, Src0TypeID,
                                        real_kind, assign_error_inexact>,
                      kernel_request_host, 1> {
      typedef typename type_of<DstTypeID>::type dst_type;
      typedef typename type_of<Src0TypeID>::type src0_type;

      void single(char *dst, char *const *src)
      {
        src0_type s = *reinterpret_cast<src0_type *>(src[0]);
        typename dst_type::value_type d;

        DYND_TRACE_ASSIGNMENT(static_cast<dst_type>(s), dst_type, s, src0_type);

#if defined(DYND_USE_FPSTATUS)
        clear_fp_status();
        d = static_cast<typename dst_type::value_type>(s);
        if (is_overflow_fp_status()) {
          std::stringstream ss;
          ss << "overflow while assigning " << ndt::make_type<src0_type>()
             << " value ";
          ss << s << " to " << ndt::make_type<dst_type>();
          throw std::overflow_error(ss.str());
        }
#else
        if (isfinite(s) &&
            (s < -std::numeric_limits<typename dst_type::value_type>::max() ||
             s > std::numeric_limits<typename dst_type::value_type>::max())) {
          std::stringstream ss;
          ss << "overflow while assigning " << ndt::make_type<src0_type>()
             << " value ";
          ss << s << " to " << ndt::make_type<dst_type>();
          throw std::overflow_error(ss.str());
        }
        d = static_cast<typename dst_type::value_type>(s);
#endif // DYND_USE_FPSTATUS

        if (d != s) {
          std::stringstream ss;
          ss << "inexact precision loss while assigning "
             << ndt::make_type<src0_type>() << " value ";
          ss << s << " to " << ndt::make_type<dst_type>();
          throw std::runtime_error(ss.str());
        }

        *reinterpret_cast<dst_type *>(dst) = d;
      }
    };

    // complex<double> -> complex<float> with overflow checking
    template <>
    struct assignment_kernel<complex_float32_type_id, complex_kind,
                             complex_float64_type_id, complex_kind,
                             assign_error_overflow>
        : base_kernel<assignment_kernel<complex_float32_type_id, complex_kind,
                                        complex_float64_type_id, complex_kind,
                                        assign_error_overflow>,
                      kernel_request_host, 1> {
      typedef complex<float> dst_type;
      typedef complex<double> src0_type;

      void single(char *dst, char *const *src)
      {
        DYND_TRACE_ASSIGNMENT(
            static_cast<complex<float>>(*reinterpret_cast<src0_type *>(src[0])),
            complex<float>, *reinterpret_cast<src0_type *>(src[0]),
            complex<double>);

#if defined(DYND_USE_FPSTATUS)
        clear_fp_status();
        *reinterpret_cast<dst_type *>(dst) =
            static_cast<complex<float>>(*reinterpret_cast<src0_type *>(src[0]));
        if (is_overflow_fp_status()) {
          std::stringstream ss;
          ss << "overflow while assigning " << ndt::make_type<complex<double>>()
             << " value ";
          ss << *src << " to " << ndt::make_type<complex<float>>();
          throw std::overflow_error(ss.str());
        }
#else
        complex<double>(s) = *reinterpret_cast<src0_type *>(src[0]);
        if (s.real() < -std::numeric_limits<float>::max() ||
            s.real() > std::numeric_limits<float>::max() ||
            s.imag() < -std::numeric_limits<float>::max() ||
            s.imag() > std::numeric_limits<float>::max()) {
          std::stringstream ss;
          ss << "overflow while assigning " << ndt::make_type<complex<double>>()
             << " value ";
          ss << s << " to " << ndt::make_type<complex<float>>();
          throw std::overflow_error(ss.str());
        }
        *reinterpret_cast<dst_type *>(dst) = static_cast<complex<float>>(s);
#endif // DYND_USE_FPSTATUS
      }
    };

    // complex<double> -> complex<float> with fractional checking
    template <>
    struct assignment_kernel<complex_float32_type_id, complex_kind,
                             complex_float64_type_id, complex_kind,
                             assign_error_fractional>
        : assignment_kernel<complex_float32_type_id, complex_kind,
                            complex_float64_type_id, complex_kind,
                            assign_error_overflow> {
    };

    // complex<double> -> complex<float> with inexact checking
    template <>
    struct assignment_kernel<complex_float32_type_id, complex_kind,
                             complex_float64_type_id, complex_kind,
                             assign_error_inexact>
        : base_kernel<assignment_kernel<complex_float32_type_id, complex_kind,
                                        complex_float64_type_id, complex_kind,
                                        assign_error_inexact>,
                      kernel_request_host, 1> {
      typedef complex<float> dst_type;
      typedef complex<double> src0_type;

      void single(char *dst, char *const *src)
      {
        DYND_TRACE_ASSIGNMENT(
            static_cast<complex<float>>(*reinterpret_cast<src0_type *>(src[0])),
            complex<float>, *reinterpret_cast<src0_type *>(src[0]),
            complex<double>);

        complex<double> s = *reinterpret_cast<src0_type *>(src[0]);
        complex<float> d;

#if defined(DYND_USE_FPSTATUS)
        clear_fp_status();
        d = static_cast<complex<float>>(s);
        if (is_overflow_fp_status()) {
          std::stringstream ss;
          ss << "overflow while assigning " << ndt::make_type<complex<double>>()
             << " value ";
          ss << *reinterpret_cast<src0_type *>(src[0]) << " to "
             << ndt::make_type<complex<float>>();
          throw std::overflow_error(ss.str());
        }
#else
        if (s.real() < -std::numeric_limits<float>::max() ||
            s.real() > std::numeric_limits<float>::max() ||
            s.imag() < -std::numeric_limits<float>::max() ||
            s.imag() > std::numeric_limits<float>::max()) {
          std::stringstream ss;
          ss << "overflow while assigning " << ndt::make_type<complex<double>>()
             << " value ";
          ss << *reinterpret_cast<src0_type *>(src[0]) << " to "
             << ndt::make_type<complex<float>>();
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
          ss << "inexact precision loss while assigning "
             << ndt::make_type<complex<double>>() << " value ";
          ss << *reinterpret_cast<src0_type *>(src[0]) << " to "
             << ndt::make_type<complex<float>>();
          throw std::runtime_error(ss.str());
        }
        *reinterpret_cast<dst_type *>(dst) = d;
      }
    };

  } // namespace dynd::nd::detail

  template <type_id_t DstTypeID, type_id_t Src0TypeID,
            assign_error_mode... ErrorMode>
  using assignment_kernel =
      detail::assignment_kernel<DstTypeID, type_kind_of<DstTypeID>::value,
                                Src0TypeID, type_kind_of<Src0TypeID>::value,
                                ErrorMode...>;

  template <type_id_t DstTypeID, type_id_t Src0TypeID>
  using assignment_virtual_kernel =
      detail::assignment_kernel<DstTypeID, type_kind_of<DstTypeID>::value,
                                Src0TypeID, type_kind_of<Src0TypeID>::value>;

  template <template <type_id_t, type_id_t, assign_error_mode...> class T>
  struct freeze_kernel {
    template <type_id_t TypeID0, type_id_t TypeID1>
    using type = T<TypeID0, TypeID1>;
  };

/*
  // Float16 -> bool
  template <>
  struct assignment_kernel<bool_type_id, bool_kind, float16_type_id,
  real_kind,
                           assign_error_nocheck>
      : base_kernel<assignment_kernel<bool_type_id, bool_kind,
  float16_type_id,
                                      real_kind, assign_error_nocheck>,
                    kernel_request_host, 1> {
    void single(char *dst, char *const *src)
    {
      // DYND_TRACE_ASSIGNMENT((bool)(!s.iszero()), bool1, s, float16);

      *reinterpret_cast<bool1 *>(dst) =
          !reinterpret_cast<float16 *>(src[0])->iszero();
    }
  };

  template <>
  struct assignment_kernel<bool_type_id, bool_kind, float16_type_id,
  real_kind,
                           assign_error_overflow>
      : base_kernel<assignment_kernel<bool_type_id, bool_kind,
  float16_type_id,
                                      real_kind, assign_error_overflow>,
                    kernel_request_host, 1> {
    void single(char *dst, char *const *src)
    {
      float tmp = float(*reinterpret_cast<float16 *>(src[0]));
      char *src_child[1] = {reinterpret_cast<char *>(&tmp)};
      assignment_kernel<bool_type_id, bool_kind, float16_type_id, real_kind,
                        assign_error_overflow>::single_wrapper(dst, src_child,
                                                               NULL);
    }
  };

  template <>
  struct assignment_kernel<bool_type_id, bool_kind, float16_type_id,
  real_kind,
                           assign_error_fractional>
      : base_kernel<assignment_kernel<bool_type_id, bool_kind,
  float16_type_id,
                                      real_kind, assign_error_fractional>,
                    kernel_request_host, 1> {
    void single(char *dst, char *const *src)
    {
      float tmp = float(*reinterpret_cast<float16 *>(src[0]));
      char *src_child[1] = {reinterpret_cast<char *>(&tmp)};
      assignment_kernel<bool_type_id, bool_kind, float16_type_id, real_kind,
                        assign_error_fractional>::single_wrapper(dst,
  src_child,
                                                                 NULL);
    }
  };

  template <>
  struct assignment_kernel<bool_type_id, bool_kind, float16_type_id,
  real_kind,
                           assign_error_inexact>
      : base_kernel<assignment_kernel<bool_type_id, bool_kind,
  float16_type_id,
                                      real_kind, assign_error_inexact>,
                    kernel_request_host, 1> {
    void single(char *dst, char *const *src)
    {
      float tmp = float(*reinterpret_cast<float16 *>(src[0]));
      char *src_child[1] = {reinterpret_cast<char *>(&tmp)};
      assignment_kernel<bool_type_id, bool_kind, float16_type_id, real_kind,
                        assign_error_inexact>::single_wrapper(dst, src_child,
                                                              NULL);
    }
  };

  // Bool -> float16
  template <>
  struct assignment_kernel<float16_type_id, real_kind, bool_type_id,
  bool_kind,
                           assign_error_nocheck>
      : base_kernel<assignment_kernel<float16_type_id, real_kind,
  bool_type_id,
                                      bool_kind, assign_error_nocheck>,
                    kernel_request_host, 1> {
    void single(char *dst, char *const *src)
    {
      // DYND_TRACE_ASSIGNMENT((bool)(!s.iszero()), bool1, s, float16);

      *reinterpret_cast<float16 *>(dst) = float16_from_bits(
          *reinterpret_cast<bool1 *>(src[0]) ? DYND_FLOAT16_ONE : 0);
    }
  };

  template <>
  struct assignment_kernel<float16_type_id, real_kind, bool_type_id,
  bool_kind,
                           assign_error_overflow>
      : assignment_kernel<float16_type_id, real_kind, bool_type_id, bool_kind,
                          assign_error_nocheck> {
  };

  template <>
  struct assignment_kernel<float16_type_id, real_kind, bool_type_id,
  bool_kind,
                           assign_error_fractional>
      : assignment_kernel<float16_type_id, real_kind, bool_type_id, bool_kind,
                          assign_error_nocheck> {
  };

  template <>
  struct assignment_kernel<float16_type_id, real_kind, bool_type_id,
  bool_kind,
                           assign_error_inexact>
      : assignment_kernel<float16_type_id, real_kind, bool_type_id, bool_kind,
                          assign_error_nocheck> {
  };

  template <type_id_t Src0TypeID, type_kind_t Src0TypeKind>
  struct assignment_kernel<float16_type_id, real_kind, Src0TypeID,
  Src0TypeKind,
                           assign_error_nocheck>
      : base_kernel<assignment_kernel<float16_type_id, real_kind, Src0TypeID,
                                      Src0TypeKind, assign_error_nocheck>,
                    kernel_request_host, 1> {
    typedef typename type_of<Src0TypeID>::type src0_type;

    void single(char *dst, char *const *src)
    {
      float tmp;
      assignment_kernel<
          float32_type_id, real_kind, Src0TypeID, Src0TypeKind,
          assign_error_nocheck>::single_wrapper(reinterpret_cast<char
  *>(&tmp),
                                                src, NULL);
      *reinterpret_cast<float16 *>(dst) = float16(tmp, assign_error_nocheck);
    }
  };

  template <type_id_t Src0TypeID, type_kind_t Src0TypeKind>
  struct assignment_kernel<float16_type_id, real_kind, Src0TypeID,
  Src0TypeKind,
                           assign_error_overflow>
      : base_kernel<assignment_kernel<float16_type_id, real_kind, Src0TypeID,
                                      Src0TypeKind, assign_error_overflow>,
                    kernel_request_host, 1> {
    typedef typename type_of<Src0TypeID>::type src0_type;

    void single(char *dst, char *const *src)
    {
      float tmp;
      assignment_kernel<
          float32_type_id, real_kind, Src0TypeID, Src0TypeKind,
          assign_error_overflow>::single_wrapper(reinterpret_cast<char
  *>(&tmp),
                                                 src, NULL);
      *reinterpret_cast<float16 *>(dst) = float16(tmp, assign_error_overflow);
    }
  };

  template <type_id_t Src0TypeID, type_kind_t Src0TypeKind>
  struct assignment_kernel<float16_type_id, real_kind, Src0TypeID,
  Src0TypeKind,
                           assign_error_fractional>
      : base_kernel<assignment_kernel<float16_type_id, real_kind, Src0TypeID,
                                      Src0TypeKind, assign_error_fractional>,
                    kernel_request_host, 1> {
    typedef typename type_of<Src0TypeID>::type src0_type;

    void single(char *dst, char *const *src)
    {
      float tmp;
      assignment_kernel<float32_type_id, real_kind, Src0TypeID, Src0TypeKind,
                        assign_error_fractional>::
          single_wrapper(reinterpret_cast<char *>(&tmp), src, NULL);
      *reinterpret_cast<float16 *>(dst) = float16(tmp,
  assign_error_fractional);
    }
  };

  template <type_id_t Src0TypeID, type_kind_t Src0TypeKind>
  struct assignment_kernel<float16_type_id, real_kind, Src0TypeID,
  Src0TypeKind,
                           assign_error_inexact>
      : base_kernel<assignment_kernel<float16_type_id, real_kind, Src0TypeID,
                                      Src0TypeKind, assign_error_inexact>,
                    kernel_request_host, 1> {
    typedef typename type_of<Src0TypeID>::type src0_type;

    void single(char *dst, char *const *src)
    {
      float tmp;
      assignment_kernel<
          float32_type_id, real_kind, Src0TypeID, Src0TypeKind,
          assign_error_inexact>::single_wrapper(reinterpret_cast<char
  *>(&tmp),
                                                src, NULL);
      *reinterpret_cast<float16 *>(dst) = float16(tmp, assign_error_inexact);
    }
  };

  template <type_id_t DstTypeID, type_kind_t DstTypeKind>
  struct assignment_kernel<DstTypeID, DstTypeKind, float16_type_id, real_kind,
                           assign_error_nocheck>
      : base_kernel<assignment_kernel<DstTypeID, DstTypeKind, float16_type_id,
                                      real_kind, assign_error_nocheck>,
                    kernel_request_host, 1> {
    void single(char *dst, char *const *src)
    {
      float tmp = static_cast<float>(*reinterpret_cast<float16 *>(src[0]));
      char *src_child[1] = {reinterpret_cast<char *>(&tmp)};
      assignment_kernel<DstTypeID, DstTypeKind, float32_type_id, real_kind,
                        assign_error_nocheck>::single_wrapper(dst, src_child,
                                                              NULL);
    }
  };

  template <type_id_t DstTypeID, type_kind_t DstTypeKind>
  struct assignment_kernel<DstTypeID, DstTypeKind, float16_type_id, real_kind,
                           assign_error_overflow>
      : base_kernel<assignment_kernel<DstTypeID, DstTypeKind, float16_type_id,
                                      real_kind, assign_error_overflow>,
                    kernel_request_host, 1> {
    void single(char *dst, char *const *src)
    {
      float tmp = static_cast<float>(*reinterpret_cast<float16 *>(src[0]));
      char *src_child[1] = {reinterpret_cast<char *>(&tmp)};
      assignment_kernel<DstTypeID, DstTypeKind, float32_type_id, real_kind,
                        assign_error_overflow>::single_wrapper(dst, src_child,
                                                              NULL);
    }
  };

  template <type_id_t DstTypeID, type_kind_t DstTypeKind>
  struct assignment_kernel<DstTypeID, DstTypeKind, float16_type_id, real_kind,
                           assign_error_fractional>
      : base_kernel<assignment_kernel<DstTypeID, DstTypeKind, float16_type_id,
                                      real_kind, assign_error_fractional>,
                    kernel_request_host, 1> {
    void single(char *dst, char *const *src)
    {
      float tmp = static_cast<float>(*reinterpret_cast<float16 *>(src[0]));
      char *src_child[1] = {reinterpret_cast<char *>(&tmp)};
      assignment_kernel<DstTypeID, DstTypeKind, float32_type_id, real_kind,
                        assign_error_fractional>::single_wrapper(dst,
  src_child,
                                                              NULL);
    }
  };

  template <type_id_t DstTypeID, type_kind_t DstTypeKind>
  struct assignment_kernel<DstTypeID, DstTypeKind, float16_type_id, real_kind,
                           assign_error_inexact>
      : base_kernel<assignment_kernel<DstTypeID, DstTypeKind, float16_type_id,
                                      real_kind, assign_error_inexact>,
                    kernel_request_host, 1> {
    void single(char *dst, char *const *src)
    {
      float tmp = static_cast<float>(*reinterpret_cast<float16 *>(src[0]));
      char *src_child[1] = {reinterpret_cast<char *>(&tmp)};
      assignment_kernel<DstTypeID, DstTypeKind, float32_type_id, real_kind,
                        assign_error_inexact>::single_wrapper(dst, src_child,
                                                              NULL);
    }
  };
*/

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

  template <class T>
  struct aligned_fixed_size_copy_assign_type
      : base_kernel<aligned_fixed_size_copy_assign_type<T>, kernel_request_host,
                    1> {
    void single(char *dst, char *const *src)
    {
      *reinterpret_cast<T *>(dst) = **reinterpret_cast<T *const *>(src);
    }

    void strided(char *dst, intptr_t dst_stride, char *const *src,
                 const intptr_t *src_stride, size_t count)
    {
      char *src0 = *src;
      intptr_t src0_stride = *src_stride;
      for (size_t i = 0; i != count; ++i) {
        *reinterpret_cast<T *>(dst) = *reinterpret_cast<T *>(src0);
        dst += dst_stride;
        src0 += src0_stride;
      }
    }
  };

  template <int N>
  struct aligned_fixed_size_copy_assign;

  template <>
  struct aligned_fixed_size_copy_assign<1>
      : base_kernel<aligned_fixed_size_copy_assign<1>, kernel_request_host, 1> {
    void single(char *dst, char *const *src) { *dst = **src; }

    void strided(char *dst, intptr_t dst_stride, char *const *src,
                 const intptr_t *src_stride, size_t count)
    {
      char *src0 = *src;
      intptr_t src0_stride = *src_stride;
      for (size_t i = 0; i != count; ++i) {
        *dst = *src0;
        dst += dst_stride;
        src0 += src0_stride;
      }
    }
  };

  template <>
  struct aligned_fixed_size_copy_assign<2>
      : aligned_fixed_size_copy_assign_type<int16_t> {
  };

  template <>
  struct aligned_fixed_size_copy_assign<4>
      : aligned_fixed_size_copy_assign_type<int32_t> {
  };

  template <>
  struct aligned_fixed_size_copy_assign<8>
      : aligned_fixed_size_copy_assign_type<int64_t> {
  };

  template <int N>
  struct unaligned_fixed_size_copy_assign
      : base_kernel<unaligned_fixed_size_copy_assign<N>, kernel_request_host,
                    1> {
    static void single(char *dst, char *const *src) { memcpy(dst, *src, N); }

    static void strided(char *dst, intptr_t dst_stride, char *const *src,
                        const intptr_t *src_stride, size_t count)
    {
      char *src0 = *src;
      intptr_t src0_stride = *src_stride;
      for (size_t i = 0; i != count; ++i) {
        memcpy(dst, src0, N);
        dst += dst_stride;
        src0 += src0_stride;
      }
    }
  };

  struct unaligned_copy_ck
      : base_kernel<unaligned_copy_ck, kernel_request_host, 1> {
    size_t data_size;

    unaligned_copy_ck(size_t data_size) : data_size(data_size) {}

    void single(char *dst, char *const *src) { memcpy(dst, *src, data_size); }

    void strided(char *dst, intptr_t dst_stride, char *const *src,
                 const intptr_t *src_stride, size_t count)
    {
      char *src0 = *src;
      intptr_t src0_stride = *src_stride;
      for (size_t i = 0; i != count; ++i) {
        memcpy(dst, src0, data_size);
        dst += dst_stride;
        src0 += src0_stride;
      }
    }
  };

  template <int N>
  struct wrap_single_as_strided_fixedcount_ck {
    static void strided(char *dst, intptr_t dst_stride, char *const *src,
                        const intptr_t *src_stride, size_t count,
                        ckernel_prefix *self)
    {
      ckernel_prefix *echild = self->get_child_ckernel(sizeof(ckernel_prefix));
      expr_single_t opchild = echild->get_function<expr_single_t>();
      char *src_copy[N];
      for (int j = 0; j < N; ++j) {
        src_copy[j] = src[j];
      }
      for (size_t i = 0; i != count; ++i) {
        opchild(dst, src_copy, echild);
        dst += dst_stride;
        for (int j = 0; j < N; ++j) {
          src_copy[j] += src_stride[j];
        }
      }
    }
  };

  template <>
  struct wrap_single_as_strided_fixedcount_ck<0> {
    static void strided(char *dst, intptr_t dst_stride,
                        char *const *DYND_UNUSED(src),
                        const intptr_t *DYND_UNUSED(src_stride), size_t count,
                        ckernel_prefix *self)
    {
      ckernel_prefix *echild = self->get_child_ckernel(sizeof(ckernel_prefix));
      expr_single_t opchild = echild->get_function<expr_single_t>();
      for (size_t i = 0; i != count; ++i) {
        opchild(dst, NULL, echild);
        dst += dst_stride;
      }
    }
  };

  struct wrap_single_as_strided_ck {
    typedef wrap_single_as_strided_ck self_type;
    ckernel_prefix base;
    intptr_t nsrc;

    static inline void strided(char *dst, intptr_t dst_stride, char *const *src,
                               const intptr_t *src_stride, size_t count,
                               ckernel_prefix *self)
    {
      intptr_t nsrc = reinterpret_cast<self_type *>(self)->nsrc;
      shortvector<char *> src_copy(nsrc, src);
      ckernel_prefix *child = self->get_child_ckernel(sizeof(self_type));
      expr_single_t child_fn = child->get_function<expr_single_t>();
      for (size_t i = 0; i != count; ++i) {
        child_fn(dst, src_copy.get(), child);
        dst += dst_stride;
        for (intptr_t j = 0; j < nsrc; ++j) {
          src_copy[j] += src_stride[j];
        }
      }
    }

    static void destruct(ckernel_prefix *self)
    {
      self->destroy_child_ckernel(sizeof(self_type));
    }
  };

} // namespace dynd::nd
} // namespace dynd