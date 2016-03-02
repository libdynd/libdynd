//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <stdexcept>

#include <dynd/fpstatus.hpp>
#include <dynd/type.hpp>
#include <dynd/math.hpp>
#include <dynd/func/assignment.hpp>
#include <dynd/kernels/cuda_launch.hpp>
#include <dynd/kernels/tuple_assignment_kernels.hpp>
#include <dynd/kernels/struct_assignment_kernels.hpp>
#include <dynd/kernels/base_kernel.hpp>
#include <dynd/eval/eval_context.hpp>
#include <dynd/types/type_id.hpp>
#include <dynd/types/categorical_type.hpp>
#include <dynd/types/char_type.hpp>
#include <dynd/types/adapt_type.hpp>
#include <dynd/types/fixed_bytes_type.hpp>
#include <dynd/types/option_type.hpp>
#include <dynd/types/fixed_string_type.hpp>
#include <dynd/parse.hpp>
#include <dynd/option.hpp>
#include <map>

#if defined(_MSC_VER)
// Tell the visual studio compiler we're accessing the FPU flags
#pragma fenv_access(on)
#endif

namespace dynd {

const inexact_check_t inexact_check = inexact_check_t();

// Trim taken from boost string algorithms library
// Trim taken from boost string algorithms library
template <typename ForwardIteratorT>
inline ForwardIteratorT trim_begin(ForwardIteratorT InBegin, ForwardIteratorT InEnd)
{
  ForwardIteratorT It = InBegin;
  for (; It != InEnd; ++It) {
    if (!isspace(*It))
      return It;
  }

  return It;
}
template <typename ForwardIteratorT>
inline ForwardIteratorT trim_end(ForwardIteratorT InBegin, ForwardIteratorT InEnd)
{
  for (ForwardIteratorT It = InEnd; It != InBegin;) {
    if (!isspace(*(--It)))
      return ++It;
  }

  return InBegin;
}
template <typename SequenceT>
inline void trim_left_if(SequenceT &Input)
{
  Input.erase(Input.begin(), trim_begin(Input.begin(), Input.end()));
}
template <typename SequenceT>
inline void trim_right_if(SequenceT &Input)
{
  Input.erase(trim_end(Input.begin(), Input.end()), Input.end());
}
template <typename SequenceT>
inline void trim(SequenceT &Input)
{
  trim_right_if(Input);
  trim_left_if(Input);
}
// End trim taken from boost string algorithms
inline void to_lower(std::string &s)
{
  for (size_t i = 0, i_end = s.size(); i != i_end; ++i) {
    s[i] = tolower(s[i]);
  }
}

namespace nd {
  namespace detail {

    template <type_id_t DstID, type_id_t DstBaseID, type_id_t Src0ID, type_id_t Src0BaseID,
              assign_error_mode... ErrorMode>
    struct assignment_kernel;

    template <type_id_t DstID, type_id_t DstBaseID, type_id_t Src0ID, type_id_t Src0BaseID,
              assign_error_mode... ErrorMode>
    struct assignment_virtual_kernel : base_kernel<assignment_virtual_kernel<DstID, DstBaseID, Src0ID, Src0BaseID>> {
      static void instantiate(char *static_data, char *data, kernel_builder *ckb, const ndt::type &dst_tp,
                              const char *dst_arrmeta, intptr_t nsrc, const ndt::type *src_tp,
                              const char *const *src_arrmeta, kernel_request_t kernreq, intptr_t nkwd,
                              const nd::array *kwds, const std::map<std::string, ndt::type> &tp_vars)
      {
        assign_error_mode error_mode = kwds[0].is_na() ? assign_error_default : kwds[0].as<assign_error_mode>();
        switch (error_mode) {
        case assign_error_default:
        case assign_error_nocheck:
          assignment_kernel<DstID, DstBaseID, Src0ID, Src0BaseID, assign_error_nocheck>::instantiate(
              static_data, data, ckb, dst_tp, dst_arrmeta, nsrc, src_tp, src_arrmeta, kernreq, nkwd, kwds, tp_vars);
          break;
        case assign_error_overflow:
          assignment_kernel<DstID, DstBaseID, Src0ID, Src0BaseID, assign_error_overflow>::instantiate(
              static_data, data, ckb, dst_tp, dst_arrmeta, nsrc, src_tp, src_arrmeta, kernreq, nkwd, kwds, tp_vars);
          break;
        case assign_error_fractional:
          assignment_kernel<DstID, DstBaseID, Src0ID, Src0BaseID, assign_error_fractional>::instantiate(
              static_data, data, ckb, dst_tp, dst_arrmeta, nsrc, src_tp, src_arrmeta, kernreq, nkwd, kwds, tp_vars);
          break;
        case assign_error_inexact:
          assignment_kernel<DstID, DstBaseID, Src0ID, Src0BaseID, assign_error_inexact>::instantiate(
              static_data, data, ckb, dst_tp, dst_arrmeta, nsrc, src_tp, src_arrmeta, kernreq, nkwd, kwds, tp_vars);
          break;
        default:
          throw std::runtime_error("error");
        }
      }
    };

    template <type_id_t DstTypeID, type_id_t DstBaseID, type_id_t Src0TypeID, type_id_t Src0BaseID,
              assign_error_mode ErrorMode>
    struct assignment_kernel<DstTypeID, DstBaseID, Src0TypeID, Src0BaseID, ErrorMode>
        : base_strided_kernel<assignment_kernel<DstTypeID, DstBaseID, Src0TypeID, Src0BaseID, ErrorMode>, 1> {
      typedef typename type_of<DstTypeID>::type dst_type;
      typedef typename type_of<Src0TypeID>::type src_type;

      void single(char *dst, char *const *src)
      {
        DYND_TRACE_ASSIGNMENT(static_cast<dst_type>(*reinterpret_cast<src_type *>(src[0])), dst_type,
                              *reinterpret_cast<src_type *>(src[0]), src_type);

        *reinterpret_cast<dst_type *>(dst) = static_cast<dst_type>(*reinterpret_cast<src_type *>(src[0]));
      }
    };

    // Complex floating point -> non-complex with no error checking
    template <type_id_t DstTypeID, type_id_t Src0TypeID>
    struct assignment_kernel<DstTypeID, int_kind_id, Src0TypeID, complex_kind_id, assign_error_nocheck>
        : base_strided_kernel<
              assignment_kernel<DstTypeID, int_kind_id, Src0TypeID, complex_kind_id, assign_error_nocheck>, 1> {
      typedef typename type_of<DstTypeID>::type dst_type;
      typedef typename type_of<Src0TypeID>::type src0_type;

      void single(char *dst, char *const *src)
      {
        src0_type s = *reinterpret_cast<src0_type *>(src[0]);

        DYND_TRACE_ASSIGNMENT(static_cast<dst_type>(s.real()), dst_type, s, src0_type);

        *reinterpret_cast<dst_type *>(dst) = static_cast<dst_type>(s.real());
      }
    };

    template <type_id_t DstTypeID, type_id_t Src0TypeID>
    struct assignment_kernel<DstTypeID, uint_kind_id, Src0TypeID, complex_kind_id, assign_error_nocheck>
        : base_strided_kernel<
              assignment_kernel<DstTypeID, uint_kind_id, Src0TypeID, complex_kind_id, assign_error_nocheck>, 1> {
      typedef typename type_of<DstTypeID>::type dst_type;
      typedef typename type_of<Src0TypeID>::type src0_type;

      void single(char *dst, char *const *src)
      {
        src0_type s = *reinterpret_cast<src0_type *>(src[0]);

        DYND_TRACE_ASSIGNMENT(static_cast<dst_type>(s.real()), dst_type, s, complex<src_real_type>);

        *reinterpret_cast<dst_type *>(dst) = static_cast<dst_type>(s.real());
      }
    };

    template <type_id_t DstTypeID, type_id_t Src0TypeID>
    struct assignment_kernel<DstTypeID, float_kind_id, Src0TypeID, complex_kind_id, assign_error_nocheck>
        : base_strided_kernel<
              assignment_kernel<DstTypeID, float_kind_id, Src0TypeID, complex_kind_id, assign_error_nocheck>, 1> {
      typedef typename type_of<DstTypeID>::type dst_type;
      typedef typename type_of<Src0TypeID>::type src0_type;

      void single(char *dst, char *const *src)
      {
        src0_type s = *reinterpret_cast<src0_type *>(src[0]);

        DYND_TRACE_ASSIGNMENT(static_cast<dst_type>(s.real()), dst_type, s, src0_type);

        *reinterpret_cast<dst_type *>(dst) = static_cast<dst_type>(s.real());
      }
    };

    // Signed int -> complex floating point with no checking
    template <type_id_t DstTypeID, type_id_t Src0TypeID>
    struct assignment_kernel<DstTypeID, complex_kind_id, Src0TypeID, int_kind_id, assign_error_nocheck>
        : base_strided_kernel<
              assignment_kernel<DstTypeID, complex_kind_id, Src0TypeID, int_kind_id, assign_error_nocheck>, 1> {
      typedef typename type_of<DstTypeID>::type dst_type;
      typedef typename type_of<Src0TypeID>::type src0_type;

      void single(char *dst, char *const *src)
      {
        src0_type s = *reinterpret_cast<src0_type *>(src[0]);

        DYND_TRACE_ASSIGNMENT(d, dst_type, s, src0_type);

        *reinterpret_cast<dst_type *>(dst) = static_cast<typename dst_type::value_type>(s);
      }
    };

    // Signed int -> complex floating point with inexact checking
    template <type_id_t DstTypeID, type_id_t Src0TypeID>
    struct assignment_kernel<DstTypeID, complex_kind_id, Src0TypeID, int_kind_id, assign_error_inexact>
        : base_strided_kernel<
              assignment_kernel<DstTypeID, complex_kind_id, Src0TypeID, int_kind_id, assign_error_inexact>, 1> {
      typedef typename type_of<DstTypeID>::type dst_type;
      typedef typename type_of<Src0TypeID>::type src0_type;

      void single(char *dst, char *const *src)
      {
        *reinterpret_cast<dst_type *>(dst) =
            check_cast<dst_type>(*reinterpret_cast<src0_type *>(src[0]), inexact_check);
      }
    };

    // Signed int -> complex floating point with other checking
    template <type_id_t DstTypeID, type_id_t Src0TypeID>
    struct assignment_kernel<DstTypeID, complex_kind_id, Src0TypeID, int_kind_id, assign_error_overflow>
        : assignment_kernel<DstTypeID, complex_kind_id, Src0TypeID, int_kind_id, assign_error_nocheck> {
    };

    template <type_id_t DstTypeID, type_id_t Src0TypeID>
    struct assignment_kernel<DstTypeID, complex_kind_id, Src0TypeID, int_kind_id, assign_error_fractional>
        : assignment_kernel<DstTypeID, complex_kind_id, Src0TypeID, int_kind_id, assign_error_nocheck> {
    };

    // Anything -> boolean with no checking
    template <type_id_t Src0TypeID, type_id_t Src0BaseTypeID>
    struct assignment_kernel<bool_id, bool_kind_id, Src0TypeID, Src0BaseTypeID, assign_error_nocheck>
        : base_strided_kernel<
              assignment_kernel<bool_id, bool_kind_id, Src0TypeID, Src0BaseTypeID, assign_error_nocheck>, 1> {
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
    struct assignment_kernel<DstTypeID, float_kind_id, Src0TypeID, uint_kind_id, assign_error_inexact>
        : base_strided_kernel<
              assignment_kernel<DstTypeID, float_kind_id, Src0TypeID, uint_kind_id, assign_error_inexact>, 1> {
      typedef typename type_of<DstTypeID>::type dst_type;
      typedef typename type_of<Src0TypeID>::type src0_type;

      void single(char *dst, char *const *src)
      {
        *reinterpret_cast<dst_type *>(dst) =
            check_cast<dst_type>(*reinterpret_cast<src0_type *>(src[0]), inexact_check);
      }
    };

    // Unsigned int -> floating point with other checking
    template <type_id_t DstTypeID, type_id_t Src0TypeID>
    struct assignment_kernel<DstTypeID, float_kind_id, Src0TypeID, uint_kind_id, assign_error_overflow>
        : assignment_kernel<DstTypeID, float_kind_id, Src0TypeID, uint_kind_id, assign_error_nocheck> {
    };

    template <type_id_t DstTypeID, type_id_t Src0TypeID>
    struct assignment_kernel<DstTypeID, float_kind_id, Src0TypeID, uint_kind_id, assign_error_fractional>
        : assignment_kernel<DstTypeID, float_kind_id, Src0TypeID, uint_kind_id, assign_error_nocheck> {
    };

    // Unsigned int -> complex floating point with no checking
    template <type_id_t DstTypeID, type_id_t Src0TypeID>
    struct assignment_kernel<DstTypeID, complex_kind_id, Src0TypeID, uint_kind_id, assign_error_nocheck>
        : base_strided_kernel<
              assignment_kernel<DstTypeID, complex_kind_id, Src0TypeID, uint_kind_id, assign_error_nocheck>, 1> {
      typedef typename type_of<DstTypeID>::type dst_type;
      typedef typename type_of<Src0TypeID>::type src0_type;

      void single(char *dst, char *const *src)
      {
        src0_type s = *reinterpret_cast<src0_type *>(src[0]);

        DYND_TRACE_ASSIGNMENT(d, dst_type, s, src0_type);

        *reinterpret_cast<dst_type *>(dst) = static_cast<typename dst_type::value_type>(s);
      }
    };

    // Unsigned int -> complex floating point with inexact checking
    template <type_id_t DstTypeID, type_id_t Src0TypeID>
    struct assignment_kernel<DstTypeID, complex_kind_id, Src0TypeID, uint_kind_id, assign_error_inexact>
        : base_strided_kernel<
              assignment_kernel<DstTypeID, complex_kind_id, Src0TypeID, uint_kind_id, assign_error_inexact>, 1> {
      typedef typename type_of<DstTypeID>::type dst_type;
      typedef typename type_of<Src0TypeID>::type src0_type;

      void single(char *dst, char *const *src)
      {
        *reinterpret_cast<dst_type *>(dst) =
            check_cast<dst_type>(*reinterpret_cast<src0_type *>(src[0]), inexact_check);
      }
    };

    // Unsigned int -> complex floating point with other checking
    template <type_id_t DstTypeID, type_id_t Src0TypeID>
    struct assignment_kernel<DstTypeID, complex_kind_id, Src0TypeID, uint_kind_id, assign_error_overflow>
        : assignment_kernel<DstTypeID, complex_kind_id, Src0TypeID, uint_kind_id, assign_error_nocheck> {
    };

    template <type_id_t DstTypeID, type_id_t Src0TypeID>
    struct assignment_kernel<DstTypeID, complex_kind_id, Src0TypeID, uint_kind_id, assign_error_fractional>
        : assignment_kernel<DstTypeID, complex_kind_id, Src0TypeID, uint_kind_id, assign_error_nocheck> {
    };

    // Floating point -> signed int with overflow checking
    template <type_id_t DstTypeID, type_id_t Src0TypeID>
    struct assignment_kernel<DstTypeID, int_kind_id, Src0TypeID, float_kind_id, assign_error_overflow>
        : base_strided_kernel<
              assignment_kernel<DstTypeID, int_kind_id, Src0TypeID, float_kind_id, assign_error_overflow>, 1> {
      typedef typename type_of<DstTypeID>::type dst_type;
      typedef typename type_of<Src0TypeID>::type src0_type;

      void single(char *dst, char *const *src)
      {
        *reinterpret_cast<dst_type *>(dst) = overflow_cast<dst_type>(*reinterpret_cast<src0_type *>(src[0]));
      }
    };

    // Floating point -> signed int with fractional checking
    template <type_id_t DstTypeID, type_id_t Src0TypeID>
    struct assignment_kernel<DstTypeID, int_kind_id, Src0TypeID, float_kind_id, assign_error_fractional>
        : base_strided_kernel<
              assignment_kernel<DstTypeID, int_kind_id, Src0TypeID, float_kind_id, assign_error_fractional>, 1> {
      typedef typename type_of<DstTypeID>::type dst_type;
      typedef typename type_of<Src0TypeID>::type src0_type;

      void single(char *dst, char *const *src)
      {
        *reinterpret_cast<dst_type *>(dst) = fractional_cast<dst_type>(*reinterpret_cast<src0_type *>(src[0]));
      }
    };

    // Floating point -> signed int with other checking
    template <type_id_t DstTypeID, type_id_t Src0TypeID>
    struct assignment_kernel<DstTypeID, int_kind_id, Src0TypeID, float_kind_id, assign_error_inexact>
        : assignment_kernel<DstTypeID, int_kind_id, Src0TypeID, float_kind_id, assign_error_fractional> {
    };

    // Complex floating point -> signed int with overflow checking
    template <type_id_t DstTypeID, type_id_t Src0TypeID>
    struct assignment_kernel<DstTypeID, int_kind_id, Src0TypeID, complex_kind_id, assign_error_overflow>
        : base_strided_kernel<
              assignment_kernel<DstTypeID, int_kind_id, Src0TypeID, complex_kind_id, assign_error_overflow>, 1> {
      typedef typename type_of<DstTypeID>::type dst_type;
      typedef typename type_of<Src0TypeID>::type src0_type;

      void single(char *dst, char *const *src)
      {
        *reinterpret_cast<dst_type *>(dst) = overflow_cast<dst_type>(*reinterpret_cast<src0_type *>(src[0]));
      }
    };

    // Complex floating point -> signed int with fractional checking
    template <type_id_t DstTypeID, type_id_t Src0TypeID>
    struct assignment_kernel<DstTypeID, int_kind_id, Src0TypeID, complex_kind_id, assign_error_fractional>
        : base_strided_kernel<
              assignment_kernel<DstTypeID, int_kind_id, Src0TypeID, complex_kind_id, assign_error_fractional>, 1> {
      typedef typename type_of<DstTypeID>::type dst_type;
      typedef typename type_of<Src0TypeID>::type src0_type;

      void single(char *dst, char *const *src)
      {
        *reinterpret_cast<dst_type *>(dst) = fractional_cast<dst_type>(*reinterpret_cast<src0_type *>(src[0]));
      }
    };

    // Complex floating point -> signed int with other checking
    template <type_id_t DstTypeID, type_id_t Src0TypeID>
    struct assignment_kernel<DstTypeID, int_kind_id, Src0TypeID, complex_kind_id, assign_error_inexact>
        : assignment_kernel<DstTypeID, int_kind_id, Src0TypeID, complex_kind_id, assign_error_fractional> {
    };

    // Floating point -> unsigned int with overflow checking
    template <type_id_t DstTypeID, type_id_t Src0TypeID>
    struct assignment_kernel<DstTypeID, uint_kind_id, Src0TypeID, float_kind_id, assign_error_overflow>
        : base_strided_kernel<
              assignment_kernel<DstTypeID, uint_kind_id, Src0TypeID, float_kind_id, assign_error_overflow>, 1> {
      typedef typename type_of<DstTypeID>::type dst_type;
      typedef typename type_of<Src0TypeID>::type src0_type;

      void single(char *dst, char *const *src)
      {
        *reinterpret_cast<dst_type *>(dst) = overflow_cast<dst_type>(*reinterpret_cast<src0_type *>(src[0]));
      }
    };

    // Floating point -> unsigned int with fractional checking
    template <type_id_t DstTypeID, type_id_t Src0TypeID>
    struct assignment_kernel<DstTypeID, uint_kind_id, Src0TypeID, float_kind_id, assign_error_fractional>
        : base_strided_kernel<
              assignment_kernel<DstTypeID, uint_kind_id, Src0TypeID, float_kind_id, assign_error_fractional>, 1> {
      typedef typename type_of<DstTypeID>::type dst_type;
      typedef typename type_of<Src0TypeID>::type src0_type;

      void single(char *dst, char *const *src)
      {
        *reinterpret_cast<dst_type *>(dst) = fractional_cast<dst_type>(*reinterpret_cast<src0_type *>(src[0]));
      }
    };

    // Floating point -> unsigned int with other checking
    template <type_id_t DstTypeID, type_id_t Src0TypeID>
    struct assignment_kernel<DstTypeID, uint_kind_id, Src0TypeID, float_kind_id, assign_error_inexact>
        : assignment_kernel<DstTypeID, uint_kind_id, Src0TypeID, float_kind_id, assign_error_fractional> {
    };

    // Complex floating point -> unsigned int with overflow checking
    template <type_id_t DstTypeID, type_id_t Src0TypeID>
    struct assignment_kernel<DstTypeID, uint_kind_id, Src0TypeID, complex_kind_id, assign_error_overflow>
        : base_strided_kernel<
              assignment_kernel<DstTypeID, uint_kind_id, Src0TypeID, complex_kind_id, assign_error_overflow>, 1> {
      typedef typename type_of<DstTypeID>::type dst_type;
      typedef typename type_of<Src0TypeID>::type src0_type;

      void single(char *dst, char *const *src)
      {
        *reinterpret_cast<dst_type *>(dst) = overflow_cast<dst_type>(*reinterpret_cast<src0_type *>(src[0]));
      }
    };

    // Complex floating point -> unsigned int with fractional checking
    template <type_id_t DstTypeID, type_id_t Src0TypeID>
    struct assignment_kernel<DstTypeID, uint_kind_id, Src0TypeID, complex_kind_id, assign_error_fractional>
        : base_strided_kernel<
              assignment_kernel<DstTypeID, uint_kind_id, Src0TypeID, complex_kind_id, assign_error_fractional>, 1> {
      typedef typename type_of<DstTypeID>::type dst_type;
      typedef typename type_of<Src0TypeID>::type src0_type;

      void single(char *dst, char *const *src)
      {
        *reinterpret_cast<dst_type *>(dst) = fractional_cast<dst_type>(*reinterpret_cast<src0_type *>(src[0]));
      }
    };

    // Complex floating point -> unsigned int with other checking
    template <type_id_t DstTypeID, type_id_t Src0TypeID>
    struct assignment_kernel<DstTypeID, uint_kind_id, Src0TypeID, complex_kind_id, assign_error_inexact>
        : assignment_kernel<DstTypeID, uint_kind_id, Src0TypeID, complex_kind_id, assign_error_fractional> {
    };

    // float -> float with no checking
    template <>
    struct assignment_kernel<float32_id, float_kind_id, float32_id, float_kind_id, assign_error_overflow>
        : assignment_kernel<float32_id, float_kind_id, float32_id, float_kind_id, assign_error_nocheck> {
    };

    template <>
    struct assignment_kernel<float32_id, float_kind_id, float32_id, float_kind_id, assign_error_fractional>
        : assignment_kernel<float32_id, float_kind_id, float32_id, float_kind_id, assign_error_nocheck> {
    };

    template <>
    struct assignment_kernel<float32_id, float_kind_id, float32_id, float_kind_id, assign_error_inexact>
        : assignment_kernel<float32_id, float_kind_id, float32_id, float_kind_id, assign_error_nocheck> {
    };

    // complex<float> -> complex<float> with no checking
    template <>
    struct assignment_kernel<complex_float32_id, complex_kind_id, complex_float32_id, complex_kind_id,
                             assign_error_overflow>
        : assignment_kernel<complex_float32_id, complex_kind_id, complex_float32_id, complex_kind_id,
                            assign_error_nocheck> {
    };

    template <>
    struct assignment_kernel<complex_float32_id, complex_kind_id, complex_float32_id, complex_kind_id,
                             assign_error_fractional>
        : assignment_kernel<complex_float32_id, complex_kind_id, complex_float32_id, complex_kind_id,
                            assign_error_nocheck> {
    };

    template <>
    struct assignment_kernel<complex_float32_id, complex_kind_id, complex_float32_id, complex_kind_id,
                             assign_error_inexact>
        : assignment_kernel<complex_float32_id, complex_kind_id, complex_float32_id, complex_kind_id,
                            assign_error_nocheck> {
    };

    // float -> double with no checking
    template <>
    struct assignment_kernel<float64_id, float_kind_id, float32_id, float_kind_id, assign_error_overflow>
        : assignment_kernel<float64_id, float_kind_id, float32_id, float_kind_id, assign_error_nocheck> {
    };

    template <>
    struct assignment_kernel<float64_id, float_kind_id, float32_id, float_kind_id, assign_error_fractional>
        : assignment_kernel<float64_id, float_kind_id, float32_id, float_kind_id, assign_error_nocheck> {
    };

    template <>
    struct assignment_kernel<float64_id, float_kind_id, float32_id, float_kind_id, assign_error_inexact>
        : assignment_kernel<float64_id, float_kind_id, float32_id, float_kind_id, assign_error_nocheck> {
    };

    // complex<float> -> complex<double> with no checking
    template <>
    struct assignment_kernel<complex_float64_id, complex_kind_id, complex_float32_id, complex_kind_id,
                             assign_error_overflow>
        : assignment_kernel<complex_float64_id, complex_kind_id, complex_float32_id, complex_kind_id,
                            assign_error_nocheck> {
    };

    template <>
    struct assignment_kernel<complex_float64_id, complex_kind_id, complex_float32_id, complex_kind_id,
                             assign_error_fractional>
        : assignment_kernel<complex_float64_id, complex_kind_id, complex_float32_id, complex_kind_id,
                            assign_error_nocheck> {
    };

    template <>
    struct assignment_kernel<complex_float64_id, complex_kind_id, complex_float32_id, complex_kind_id,
                             assign_error_inexact>
        : assignment_kernel<complex_float64_id, complex_kind_id, complex_float32_id, complex_kind_id,
                            assign_error_nocheck> {
    };

    // double -> double with no checking
    template <>
    struct assignment_kernel<float64_id, float_kind_id, float64_id, float_kind_id, assign_error_overflow>
        : assignment_kernel<float64_id, float_kind_id, float64_id, float_kind_id, assign_error_nocheck> {
    };

    template <>
    struct assignment_kernel<float64_id, float_kind_id, float64_id, float_kind_id, assign_error_fractional>
        : assignment_kernel<float64_id, float_kind_id, float64_id, float_kind_id, assign_error_nocheck> {
    };

    template <>
    struct assignment_kernel<float64_id, float_kind_id, float64_id, float_kind_id, assign_error_inexact>
        : assignment_kernel<float64_id, float_kind_id, float64_id, float_kind_id, assign_error_nocheck> {
    };

    // complex<double> -> complex<double> with no checking
    template <>
    struct assignment_kernel<complex_float64_id, complex_kind_id, complex_float64_id, complex_kind_id,
                             assign_error_overflow>
        : assignment_kernel<complex_float64_id, complex_kind_id, complex_float64_id, complex_kind_id,
                            assign_error_nocheck> {
    };

    template <>
    struct assignment_kernel<complex_float64_id, complex_kind_id, complex_float64_id, complex_kind_id,
                             assign_error_fractional>
        : assignment_kernel<complex_float64_id, complex_kind_id, complex_float64_id, complex_kind_id,
                            assign_error_nocheck> {
    };

    template <>
    struct assignment_kernel<complex_float64_id, complex_kind_id, complex_float64_id, complex_kind_id,
                             assign_error_inexact>
        : assignment_kernel<complex_float64_id, complex_kind_id, complex_float64_id, complex_kind_id,
                            assign_error_nocheck> {
    };

    // real -> real with overflow checking
    template <type_id_t DstTypeID, type_id_t Src0TypeID>
    struct assignment_kernel<DstTypeID, float_kind_id, Src0TypeID, float_kind_id, assign_error_overflow>
        : base_strided_kernel<
              assignment_kernel<DstTypeID, float_kind_id, Src0TypeID, float_kind_id, assign_error_overflow>, 1> {
      typedef typename type_of<DstTypeID>::type dst_type;
      typedef typename type_of<Src0TypeID>::type src0_type;

      void single(char *dst, char *const *src)
      {
        *reinterpret_cast<dst_type *>(dst) = overflow_cast<dst_type>(*reinterpret_cast<src0_type *>(src[0]));
      }
    };

    // real -> real with fractional checking
    template <type_id_t DstTypeID, type_id_t Src0TypeID>
    struct assignment_kernel<DstTypeID, float_kind_id, Src0TypeID, float_kind_id, assign_error_fractional>
        : assignment_kernel<DstTypeID, float_kind_id, Src0TypeID, float_kind_id, assign_error_overflow> {
    };

    // real -> real with inexact checking
    template <type_id_t DstTypeID, type_id_t Src0TypeID>
    struct assignment_kernel<DstTypeID, float_kind_id, Src0TypeID, float_kind_id, assign_error_inexact>
        : base_strided_kernel<
              assignment_kernel<DstTypeID, float_kind_id, Src0TypeID, float_kind_id, assign_error_inexact>, 1> {
      typedef typename type_of<DstTypeID>::type dst_type;
      typedef typename type_of<Src0TypeID>::type src0_type;

      void single(char *dst, char *const *src)
      {
        *reinterpret_cast<dst_type *>(dst) =
            check_cast<dst_type>(*reinterpret_cast<src0_type *>(src[0]), inexact_check);
      }
    };

    // Anything -> boolean with overflow checking
    template <type_id_t Src0TypeID, type_id_t Src0BaseTypeID>
    struct assignment_kernel<bool_id, bool_kind_id, Src0TypeID, Src0BaseTypeID, assign_error_overflow>
        : base_strided_kernel<
              assignment_kernel<bool_id, bool_kind_id, Src0TypeID, Src0BaseTypeID, assign_error_overflow>, 1> {
      typedef typename type_of<Src0TypeID>::type src0_type;

      void single(char *dst, char *const *src)
      {
        *reinterpret_cast<bool1 *>(dst) = overflow_cast<bool1>(*reinterpret_cast<src0_type *>(src[0]));
      }
    };

    // Anything -> boolean with other error checking
    template <type_id_t Src0TypeID, type_id_t Src0BaseTypeID>
    struct assignment_kernel<bool_id, bool_kind_id, Src0TypeID, Src0BaseTypeID, assign_error_fractional>
        : assignment_kernel<bool_id, bool_kind_id, Src0TypeID, Src0BaseTypeID, assign_error_overflow> {
    };

    template <type_id_t Src0TypeID, type_id_t Src0BaseTypeID>
    struct assignment_kernel<bool_id, bool_kind_id, Src0TypeID, Src0BaseTypeID, assign_error_inexact>
        : assignment_kernel<bool_id, bool_kind_id, Src0TypeID, Src0BaseTypeID, assign_error_overflow> {
    };

    // Boolean -> boolean with other error checking
    template <>
    struct assignment_kernel<bool_id, bool_kind_id, bool_id, bool_kind_id, assign_error_overflow>
        : assignment_kernel<bool_id, bool_kind_id, bool_id, bool_kind_id, assign_error_nocheck> {
    };

    template <>
    struct assignment_kernel<bool_id, bool_kind_id, bool_id, bool_kind_id, assign_error_fractional>
        : assignment_kernel<bool_id, bool_kind_id, bool_id, bool_kind_id, assign_error_nocheck> {
    };

    template <>
    struct assignment_kernel<bool_id, bool_kind_id, bool_id, bool_kind_id, assign_error_inexact>
        : assignment_kernel<bool_id, bool_kind_id, bool_id, bool_kind_id, assign_error_nocheck> {
    };

    // Boolean -> anything with other error checking
    template <type_id_t DstTypeID, type_id_t DstBaseID>
    struct assignment_kernel<DstTypeID, DstBaseID, bool_id, bool_kind_id, assign_error_overflow>
        : assignment_kernel<DstTypeID, DstBaseID, bool_id, bool_kind_id, assign_error_nocheck> {
    };

    template <type_id_t DstTypeID, type_id_t DstBaseID>
    struct assignment_kernel<DstTypeID, DstBaseID, bool_id, bool_kind_id, assign_error_fractional>
        : assignment_kernel<DstTypeID, DstBaseID, bool_id, bool_kind_id, assign_error_nocheck> {
    };

    template <type_id_t DstTypeID, type_id_t DstBaseID>
    struct assignment_kernel<DstTypeID, DstBaseID, bool_id, bool_kind_id, assign_error_inexact>
        : assignment_kernel<DstTypeID, DstBaseID, bool_id, bool_kind_id, assign_error_nocheck> {
    };

    // Signed int -> signed int with overflow checking
    template <type_id_t DstTypeID, type_id_t Src0TypeID>
    struct assignment_kernel<DstTypeID, int_kind_id, Src0TypeID, int_kind_id, assign_error_overflow>
        : base_strided_kernel<assignment_kernel<DstTypeID, int_kind_id, Src0TypeID, int_kind_id, assign_error_overflow>,
                              1> {
      typedef typename type_of<DstTypeID>::type dst_type;
      typedef typename type_of<Src0TypeID>::type src0_type;

      void single(char *dst, char *const *src)
      {
        *reinterpret_cast<dst_type *>(dst) = overflow_cast<dst_type>(*reinterpret_cast<src0_type *>(src[0]));
      }
    };

    // Signed int -> signed int with other error checking
    template <type_id_t DstTypeID, type_id_t Src0TypeID>
    struct assignment_kernel<DstTypeID, int_kind_id, Src0TypeID, int_kind_id, assign_error_fractional>
        : assignment_kernel<DstTypeID, int_kind_id, Src0TypeID, int_kind_id, assign_error_overflow> {
    };

    template <type_id_t DstTypeID, type_id_t Src0TypeID>
    struct assignment_kernel<DstTypeID, int_kind_id, Src0TypeID, int_kind_id, assign_error_inexact>
        : assignment_kernel<DstTypeID, int_kind_id, Src0TypeID, int_kind_id, assign_error_overflow> {
    };

    // Unsigned int -> signed int with overflow checking just when sizeof(dst) <= sizeof(src)
    template <type_id_t DstTypeID, type_id_t Src0TypeID>
    struct assignment_kernel<DstTypeID, int_kind_id, Src0TypeID, uint_kind_id, assign_error_overflow>
        : base_strided_kernel<
              assignment_kernel<DstTypeID, int_kind_id, Src0TypeID, uint_kind_id, assign_error_overflow>, 1> {
      typedef typename type_of<DstTypeID>::type dst_type;
      typedef typename type_of<Src0TypeID>::type src0_type;

      void single(char *dst, char *const *src)
      {
        *reinterpret_cast<dst_type *>(dst) = overflow_cast<dst_type>(*reinterpret_cast<src0_type *>(src[0]));
      }
    };

    // Unsigned int -> signed int with other error checking
    template <type_id_t DstTypeID, type_id_t Src0TypeID>
    struct assignment_kernel<DstTypeID, int_kind_id, Src0TypeID, uint_kind_id, assign_error_fractional>
        : assignment_kernel<DstTypeID, int_kind_id, Src0TypeID, uint_kind_id, assign_error_overflow> {
    };

    template <type_id_t DstTypeID, type_id_t Src0TypeID>
    struct assignment_kernel<DstTypeID, int_kind_id, Src0TypeID, uint_kind_id, assign_error_inexact>
        : assignment_kernel<DstTypeID, int_kind_id, Src0TypeID, uint_kind_id, assign_error_overflow> {
    };

    // Signed int -> unsigned int with positive overflow checking just when sizeof(dst) < sizeof(src)
    template <type_id_t DstTypeID, type_id_t Src0TypeID>
    struct assignment_kernel<DstTypeID, uint_kind_id, Src0TypeID, int_kind_id, assign_error_overflow>
        : base_strided_kernel<
              assignment_kernel<DstTypeID, uint_kind_id, Src0TypeID, int_kind_id, assign_error_overflow>, 1> {
      typedef typename type_of<DstTypeID>::type dst_type;
      typedef typename type_of<Src0TypeID>::type src0_type;

      void single(char *dst, char *const *src)
      {
        *reinterpret_cast<dst_type *>(dst) = overflow_cast<dst_type>(*reinterpret_cast<src0_type *>(src[0]));
      }
    };

    // Signed int -> unsigned int with other error checking
    template <type_id_t DstTypeID, type_id_t Src0TypeID>
    struct assignment_kernel<DstTypeID, uint_kind_id, Src0TypeID, int_kind_id, assign_error_fractional>
        : assignment_kernel<DstTypeID, uint_kind_id, Src0TypeID, int_kind_id, assign_error_overflow> {
    };

    template <type_id_t DstTypeID, type_id_t Src0TypeID>
    struct assignment_kernel<DstTypeID, uint_kind_id, Src0TypeID, int_kind_id, assign_error_inexact>
        : assignment_kernel<DstTypeID, uint_kind_id, Src0TypeID, int_kind_id, assign_error_overflow> {
    };

    // Unsigned int -> unsigned int with overflow checking just when sizeof(dst) < sizeof(src)
    template <type_id_t DstTypeID, type_id_t Src0TypeID>
    struct assignment_kernel<DstTypeID, uint_kind_id, Src0TypeID, uint_kind_id, assign_error_overflow>
        : base_strided_kernel<
              assignment_kernel<DstTypeID, uint_kind_id, Src0TypeID, uint_kind_id, assign_error_overflow>, 1> {
      typedef typename type_of<DstTypeID>::type dst_type;
      typedef typename type_of<Src0TypeID>::type src0_type;

      void single(char *dst, char *const *src)
      {
        *reinterpret_cast<dst_type *>(dst) = overflow_cast<dst_type>(*reinterpret_cast<src0_type *>(src[0]));
      }
    };

    // Unsigned int -> unsigned int with other error checking
    template <type_id_t DstTypeID, type_id_t Src0TypeID>
    struct assignment_kernel<DstTypeID, uint_kind_id, Src0TypeID, uint_kind_id, assign_error_fractional>
        : assignment_kernel<DstTypeID, uint_kind_id, Src0TypeID, uint_kind_id, assign_error_overflow> {
    };

    template <type_id_t DstTypeID, type_id_t Src0TypeID>
    struct assignment_kernel<DstTypeID, uint_kind_id, Src0TypeID, uint_kind_id, assign_error_inexact>
        : assignment_kernel<DstTypeID, uint_kind_id, Src0TypeID, uint_kind_id, assign_error_overflow> {
    };

    // Signed int -> floating point with inexact checking
    template <type_id_t DstTypeID, type_id_t Src0TypeID>
    struct assignment_kernel<DstTypeID, float_kind_id, Src0TypeID, int_kind_id, assign_error_inexact>
        : base_strided_kernel<
              assignment_kernel<DstTypeID, float_kind_id, Src0TypeID, int_kind_id, assign_error_inexact>, 1> {
      typedef typename type_of<DstTypeID>::type dst_type;
      typedef typename type_of<Src0TypeID>::type src0_type;

      void single(char *dst, char *const *src)
      {
        *reinterpret_cast<dst_type *>(dst) =
            check_cast<dst_type>(*reinterpret_cast<src0_type *>(src[0]), inexact_check);
      }
    };

    // Signed int -> floating point with other checking
    template <type_id_t DstTypeID, type_id_t Src0TypeID>
    struct assignment_kernel<DstTypeID, float_kind_id, Src0TypeID, int_kind_id, assign_error_overflow>
        : assignment_kernel<DstTypeID, float_kind_id, Src0TypeID, int_kind_id, assign_error_nocheck> {
    };

    template <type_id_t DstTypeID, type_id_t Src0TypeID>
    struct assignment_kernel<DstTypeID, float_kind_id, Src0TypeID, int_kind_id, assign_error_fractional>
        : assignment_kernel<DstTypeID, float_kind_id, Src0TypeID, int_kind_id, assign_error_nocheck> {
    };

    template <type_id_t DstTypeID, type_id_t Src0TypeID, assign_error_mode ErrorMode>
    struct assignment_kernel<DstTypeID, complex_kind_id, Src0TypeID, float_kind_id, ErrorMode>
        : base_strided_kernel<assignment_kernel<DstTypeID, complex_kind_id, Src0TypeID, float_kind_id, ErrorMode>, 1> {
      typedef typename type_of<DstTypeID>::type dst_type;
      typedef typename type_of<Src0TypeID>::type src0_type;

      void single(char *dst, char *const *src)
      {
        src0_type s = *reinterpret_cast<src0_type *>(src[0]);

        DYND_TRACE_ASSIGNMENT(static_cast<dst_type>(s), dst_type, s, src0_type);

        *reinterpret_cast<dst_type *>(dst) = static_cast<typename dst_type::value_type>(s);
      }
    };

    // complex -> real with overflow checking
    template <type_id_t DstTypeID, type_id_t Src0TypeID>
    struct assignment_kernel<DstTypeID, float_kind_id, Src0TypeID, complex_kind_id, assign_error_overflow>
        : base_strided_kernel<
              assignment_kernel<DstTypeID, float_kind_id, Src0TypeID, complex_kind_id, assign_error_overflow>, 1> {
      typedef typename type_of<DstTypeID>::type dst_type;
      typedef typename type_of<Src0TypeID>::type src0_type;

      void single(char *dst, char *const *src)
      {
        *reinterpret_cast<dst_type *>(dst) = overflow_cast<dst_type>(*reinterpret_cast<src0_type *>(src[0]));
      }
    };

    // complex -> real with inexact checking
    template <type_id_t DstTypeID, type_id_t Src0TypeID>
    struct assignment_kernel<DstTypeID, float_kind_id, Src0TypeID, complex_kind_id, assign_error_inexact>
        : assignment_kernel<DstTypeID, float_kind_id, Src0TypeID, complex_kind_id, assign_error_overflow> {
    };

    // complex -> real with fractional checking
    template <type_id_t DstTypeID, type_id_t SrcTypeID>
    struct assignment_kernel<DstTypeID, float_kind_id, SrcTypeID, complex_kind_id, assign_error_fractional>
        : assignment_kernel<DstTypeID, float_kind_id, SrcTypeID, complex_kind_id, assign_error_overflow> {
    };

    // complex<double> -> float with inexact checking
    template <>
    struct assignment_kernel<float32_id, float_kind_id, complex_float64_id, complex_kind_id, assign_error_inexact>
        : base_strided_kernel<
              assignment_kernel<float32_id, float_kind_id, complex_float64_id, complex_kind_id, assign_error_inexact>,
              1> {
      void single(char *dst, char *const *src)
      {
        *reinterpret_cast<float *>(dst) =
            check_cast<float>(*reinterpret_cast<complex<double> *>(src[0]), inexact_check);
      }
    };

    // real -> complex with overflow checking
    template <type_id_t DstTypeID, type_id_t Src0TypeID>
    struct assignment_kernel<DstTypeID, complex_kind_id, Src0TypeID, float_kind_id, assign_error_overflow>
        : base_strided_kernel<
              assignment_kernel<DstTypeID, complex_kind_id, Src0TypeID, float_kind_id, assign_error_overflow>, 1> {
      typedef typename type_of<DstTypeID>::type dst_type;
      typedef typename type_of<Src0TypeID>::type src0_type;

      void single(char *dst, char *const *src)
      {
        *reinterpret_cast<dst_type *>(dst) = overflow_cast<dst_type>(*reinterpret_cast<src0_type *>(src[0]));
      }
    };

    // real -> complex with fractional checking
    template <type_id_t DstTypeID, type_id_t Src0TypeID>
    struct assignment_kernel<DstTypeID, complex_kind_id, Src0TypeID, float_kind_id, assign_error_fractional>
        : assignment_kernel<DstTypeID, complex_kind_id, Src0TypeID, float_kind_id, assign_error_overflow> {
    };

    // real -> complex with inexact checking
    template <type_id_t DstTypeID, type_id_t Src0TypeID>
    struct assignment_kernel<DstTypeID, complex_kind_id, Src0TypeID, float_kind_id, assign_error_inexact>
        : base_strided_kernel<
              assignment_kernel<DstTypeID, complex_kind_id, Src0TypeID, float_kind_id, assign_error_inexact>, 1> {
      typedef typename type_of<DstTypeID>::type dst_type;
      typedef typename type_of<Src0TypeID>::type src0_type;

      void single(char *dst, char *const *src)
      {
        *reinterpret_cast<dst_type *>(dst) =
            check_cast<dst_type>(*reinterpret_cast<src0_type *>(src[0]), inexact_check);
      }
    };

    // complex<double> -> complex<float> with overflow checking
    template <>
    struct assignment_kernel<complex_float32_id, complex_kind_id, complex_float64_id, complex_kind_id,
                             assign_error_overflow>
        : base_strided_kernel<assignment_kernel<complex_float32_id, complex_kind_id, complex_float64_id,
                                                complex_kind_id, assign_error_overflow>,
                              1> {
      typedef complex<float> dst_type;
      typedef complex<double> src0_type;

      void single(char *dst, char *const *src)
      {
        *reinterpret_cast<dst_type *>(dst) = overflow_cast<dst_type>(*reinterpret_cast<src0_type *>(src[0]));
      }
    };

    // complex<double> -> complex<float> with fractional checking
    template <>
    struct assignment_kernel<complex_float32_id, complex_kind_id, complex_float64_id, complex_kind_id,
                             assign_error_fractional>
        : assignment_kernel<complex_float32_id, complex_kind_id, complex_float64_id, complex_kind_id,
                            assign_error_overflow> {
    };

    // complex<double> -> complex<float> with inexact checking
    template <>
    struct assignment_kernel<complex_float32_id, complex_kind_id, complex_float64_id, complex_kind_id,
                             assign_error_inexact>
        : base_strided_kernel<assignment_kernel<complex_float32_id, complex_kind_id, complex_float64_id,
                                                complex_kind_id, assign_error_inexact>,
                              1> {
      typedef complex<float> dst_type;
      typedef complex<double> src0_type;

      void single(char *dst, char *const *src)
      {
        *reinterpret_cast<dst_type *>(dst) =
            check_cast<dst_type>(*reinterpret_cast<src0_type *>(src[0]), inexact_check);
      }
    };

    /**
     * A ckernel which assigns option[S] to option[T].
     */
    template <assign_error_mode ErrorMode>
    struct assignment_kernel<option_id, any_kind_id, option_id, any_kind_id, ErrorMode>
        : base_strided_kernel<assignment_kernel<option_id, any_kind_id, option_id, any_kind_id, ErrorMode>, 1> {
      // The default child is the src is_avail ckernel
      // This child is the dst assign_na ckernel
      size_t m_dst_assign_na_offset;
      size_t m_value_assign_offset;

      ~assignment_kernel()
      {
        // src_is_avail
        this->get_child()->destroy();
        // dst_assign_na
        this->get_child(m_dst_assign_na_offset)->destroy();
        // value_assign
        this->get_child(m_value_assign_offset)->destroy();
      }

      void single(char *dst, char *const *src)
      {
        // Check whether the value is available
        // TODO: Would be nice to do this as a predicate
        //       instead of having to go through a dst pointer
        kernel_prefix *src_is_na = this->get_child();
        kernel_single_t src_is_na_fn = src_is_na->get_function<kernel_single_t>();
        bool1 missing = bool1(false);
        src_is_na_fn(src_is_na, reinterpret_cast<char *>(&missing), src);
        if (!missing) {
          // It's available, copy using value assignment
          kernel_prefix *value_assign = this->get_child(m_value_assign_offset);
          kernel_single_t value_assign_fn = value_assign->get_function<kernel_single_t>();
          value_assign_fn(value_assign, dst, src);
        }
        else {
          // It's not available, assign an NA
          kernel_prefix *dst_assign_na = this->get_child(m_dst_assign_na_offset);
          kernel_single_t dst_assign_na_fn = dst_assign_na->get_function<kernel_single_t>();
          dst_assign_na_fn(dst_assign_na, dst, NULL);
        }
      }

      void strided(char *dst, intptr_t dst_stride, char *const *src, const intptr_t *src_stride, size_t count)
      {
        // Three child ckernels
        kernel_prefix *src_is_na = this->get_child();
        kernel_strided_t src_is_na_fn = src_is_na->get_function<kernel_strided_t>();
        kernel_prefix *value_assign = this->get_child(m_value_assign_offset);
        kernel_strided_t value_assign_fn = value_assign->get_function<kernel_strided_t>();
        kernel_prefix *dst_assign_na = this->get_child(m_dst_assign_na_offset);
        kernel_strided_t dst_assign_na_fn = dst_assign_na->get_function<kernel_strided_t>();
        // Process in chunks using the dynd default buffer size
        bool1 missing[DYND_BUFFER_CHUNK_SIZE];
        while (count > 0) {
          size_t chunk_size = std::min(count, (size_t)DYND_BUFFER_CHUNK_SIZE);
          count -= chunk_size;
          src_is_na_fn(src_is_na, reinterpret_cast<char *>(missing), 1, src, src_stride, chunk_size);
          void *missing_ptr = missing;
          char *src_copy = src[0];
          do {
            // Process a run of available values
            void *next_missing_ptr = memchr(missing_ptr, 1, chunk_size);
            if (!next_missing_ptr) {
              value_assign_fn(value_assign, dst, dst_stride, &src_copy, src_stride, chunk_size);
              dst += chunk_size * dst_stride;
              src += chunk_size * src_stride[0];
              break;
            }
            else if (next_missing_ptr > missing_ptr) {
              size_t segment_size = (char *)next_missing_ptr - (char *)missing_ptr;
              value_assign_fn(value_assign, dst, dst_stride, &src_copy, src_stride, segment_size);
              dst += segment_size * dst_stride;
              src_copy += segment_size * src_stride[0];
              chunk_size -= segment_size;
              missing_ptr = next_missing_ptr;
            }
            // Process a run of not available values
            next_missing_ptr = memchr(missing_ptr, 0, chunk_size);
            if (!next_missing_ptr) {
              dst_assign_na_fn(dst_assign_na, dst, dst_stride, NULL, NULL, chunk_size);
              dst += chunk_size * dst_stride;
              src_copy += chunk_size * src_stride[0];
              break;
            }
            else if (next_missing_ptr > missing_ptr) {
              size_t segment_size = (char *)next_missing_ptr - (char *)missing_ptr;
              dst_assign_na_fn(dst_assign_na, dst, dst_stride, NULL, NULL, segment_size);
              dst += segment_size * dst_stride;
              src_copy += segment_size * src_stride[0];
              chunk_size -= segment_size;
              missing_ptr = next_missing_ptr;
            }
          } while (chunk_size > 0);
        }
      }

      static void instantiate(char *DYND_UNUSED(static_data), char *DYND_UNUSED(data), kernel_builder *ckb,
                              const ndt::type &dst_tp, const char *dst_arrmeta, intptr_t nsrc, const ndt::type *src_tp,
                              const char *const *src_arrmeta, kernel_request_t kernreq, intptr_t nkwd,
                              const nd::array *kwds, const std::map<std::string, ndt::type> &tp_vars)
      {
        intptr_t ckb_offset = ckb->size();
        intptr_t root_ckb_offset = ckb_offset;
        typedef assignment_kernel self_type;
        if (dst_tp.get_id() != option_id || src_tp[0].get_id() != option_id) {
          std::stringstream ss;
          ss << "option to option kernel needs option types, got " << dst_tp << " and " << src_tp[0];
          throw std::invalid_argument(ss.str());
        }
        const ndt::type &dst_val_tp = dst_tp.extended<ndt::option_type>()->get_value_type();
        const ndt::type &src_val_tp = src_tp[0].extended<ndt::option_type>()->get_value_type();
        ckb->emplace_back<self_type>(kernreq);
        ckb_offset = ckb->size();
        // instantiate src_is_avail
        nd::callable &is_na = nd::is_na::get();
        is_na.get()->instantiate(is_na->static_data(), NULL, ckb, ndt::make_type<bool1>(), NULL, nsrc, src_tp,
                                 src_arrmeta, kernreq | kernel_request_data_only, nkwd, kwds, tp_vars);
        ckb_offset = ckb->size();
        // instantiate dst_assign_na
        ckb->reserve(ckb_offset + sizeof(kernel_prefix));
        self_type *self = ckb->get_at<self_type>(root_ckb_offset);
        self->m_dst_assign_na_offset = ckb_offset - root_ckb_offset;
        nd::callable &assign_na = nd::assign_na::get();
        assign_na.get()->instantiate(assign_na->static_data(), NULL, ckb, dst_tp, dst_arrmeta, nsrc, NULL, NULL,
                                     kernreq | kernel_request_data_only, nkwd, kwds, tp_vars);
        ckb_offset = ckb->size();
        // instantiate value_assign
        ckb->reserve(ckb_offset + sizeof(kernel_prefix));
        self = ckb->get_at<self_type>(root_ckb_offset);
        self->m_value_assign_offset = ckb_offset - root_ckb_offset;
        assign::get()->instantiate(nd::assign::get()->static_data(), NULL, ckb, dst_val_tp, dst_arrmeta, 1, &src_val_tp,
                                   src_arrmeta, kernreq | kernel_request_data_only, nkwd, kwds, tp_vars);
      }
    };

    struct DYND_API string_to_option_bool_ck : nd::base_strided_kernel<string_to_option_bool_ck, 1> {
      void single(char *dst, char *const *src)
      {
        const string *std = reinterpret_cast<string *>(src[0]);
        *reinterpret_cast<bool1 *>(dst) = parse<bool>(std->begin(), std->end());
      }
    };

    struct DYND_API string_to_option_number_ck : nd::base_strided_kernel<string_to_option_number_ck, 1> {
      type_id_t m_tid;
      assign_error_mode m_errmode;

      string_to_option_number_ck() {}

      string_to_option_number_ck(type_id_t tid, assign_error_mode errmode) : m_tid(tid), m_errmode(errmode) {}

      void single(char *dst, char *const *src)
      {
        const string *std = reinterpret_cast<string *>(src[0]);
        string_to_number(dst, m_tid, std->begin(), std->end(), m_errmode);
      }

      static void instantiate(char *DYND_UNUSED(static_data), char *DYND_UNUSED(data), kernel_builder *ckb,
                              const ndt::type &dst_tp, const char *DYND_UNUSED(dst_arrmeta), intptr_t DYND_UNUSED(nsrc),
                              const ndt::type *DYND_UNUSED(src_tp), const char *const *DYND_UNUSED(src_arrmeta),
                              kernel_request_t kernreq, intptr_t DYND_UNUSED(nkwd), const array *kwds,
                              const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars))
      {
        ckb->emplace_back<string_to_option_number_ck>(kernreq, dst_tp.get_id(), kwds[0].as<assign_error_mode>());
      }
    };

    struct DYND_API string_to_option_tp_ck : nd::base_strided_kernel<string_to_option_tp_ck, 1> {
      intptr_t m_dst_assign_na_offset;

      ~string_to_option_tp_ck()
      {
        // value_assign
        get_child()->destroy();
        // dst_assign_na
        get_child(m_dst_assign_na_offset)->destroy();
      }

      void single(char *dst, char *const *src)
      {
        const string *std = reinterpret_cast<string *>(src[0]);
        if (parse_na(std->begin(), std->end())) {
          // It's not available, assign an NA
          kernel_prefix *dst_assign_na = get_child(m_dst_assign_na_offset);
          kernel_single_t dst_assign_na_fn = dst_assign_na->get_function<kernel_single_t>();
          dst_assign_na_fn(dst_assign_na, dst, NULL);
        }
        else {
          // It's available, copy using value assignment
          kernel_prefix *value_assign = get_child();
          kernel_single_t value_assign_fn = value_assign->get_function<kernel_single_t>();
          value_assign_fn(value_assign, dst, src);
        }
      }
    };

    template <type_id_t Src0TypeID, assign_error_mode ErrorMode>
    struct assignment_kernel<option_id, any_kind_id, Src0TypeID, float_kind_id, ErrorMode> {
      static void instantiate(char *DYND_UNUSED(static_data), char *DYND_UNUSED(data), kernel_builder *ckb,
                              const ndt::type &dst_tp, const char *dst_arrmeta, intptr_t nsrc, const ndt::type *src_tp,
                              const char *const *src_arrmeta, kernel_request_t kernreq, intptr_t nkwd,
                              const nd::array *kwds, const std::map<std::string, ndt::type> &tp_vars)
      {
        // Deal with some float32 to option[T] conversions where any NaN is
        // interpreted
        // as NA.
        ndt::type src_tp_as_option = ndt::make_type<ndt::option_type>(src_tp[0]);
        assignment_kernel<option_id, any_kind_id, option_id, any_kind_id, ErrorMode>::instantiate(
            NULL, NULL, ckb, dst_tp, dst_arrmeta, nsrc, &src_tp_as_option, src_arrmeta, kernreq, nkwd, kwds, tp_vars);
      }
    };

    template <assign_error_mode ErrorMode>
    struct assignment_kernel<option_id, any_kind_id, string_id, string_kind_id, ErrorMode> {
      static void instantiate(char *DYND_UNUSED(static_data), char *DYND_UNUSED(data), kernel_builder *ckb,
                              const ndt::type &dst_tp, const char *dst_arrmeta, intptr_t nsrc, const ndt::type *src_tp,
                              const char *const *src_arrmeta, kernel_request_t kernreq, intptr_t nkwd,
                              const nd::array *kwds, const std::map<std::string, ndt::type> &tp_vars)
      {
        // Deal with some string to option[T] conversions where string values
        // might mean NA
        if (dst_tp.get_id() != option_id ||
            !(src_tp[0].get_base_id() == string_kind_id ||
              (src_tp[0].get_id() == option_id &&
               src_tp[0].extended<ndt::option_type>()->get_value_type().get_base_id() == string_kind_id))) {
          std::stringstream ss;
          ss << "string to option kernel needs string/option types, got (" << src_tp[0] << ") -> " << dst_tp;
          throw std::invalid_argument(ss.str());
        }

        type_id_t tid = dst_tp.extended<ndt::option_type>()->get_value_type().get_id();
        switch (tid) {
        case bool_id:
          ckb->emplace_back<string_to_option_bool_ck>(kernreq);
          return;
        case int8_id:
        case int16_id:
        case int32_id:
        case int64_id:
        case int128_id:
        case float16_id:
        case float32_id:
        case float64_id:
          ckb->emplace_back<string_to_option_number_ck>(kernreq, tid, ErrorMode);
          return;
        case string_id: {
          // Just a string to string assignment
          assign::get()->instantiate(nd::assign::get()->static_data(), NULL, ckb,
                                     dst_tp.extended<ndt::option_type>()->get_value_type(), dst_arrmeta, nsrc, src_tp,
                                     src_arrmeta, kernreq, nkwd, kwds, tp_vars);
          return;
        }
        default:
          break;
        }

        // Fall back to an adaptor that checks for a few standard
        // missing value tokens, then uses the standard value assignment
        intptr_t ckb_offset = ckb->size();
        intptr_t root_ckb_offset = ckb_offset;
        ckb->emplace_back<string_to_option_tp_ck>(kernreq);
        ckb_offset = ckb->size();
        // First child ckernel is the value assignment
        assign::get()->instantiate(assign::get()->static_data(), NULL, ckb,
                                   dst_tp.extended<ndt::option_type>()->get_value_type(), dst_arrmeta, nsrc, src_tp,
                                   src_arrmeta, kernreq | kernel_request_data_only, nkwd, kwds, tp_vars);
        ckb_offset = ckb->size();
        // Re-acquire self because the address may have changed
        string_to_option_tp_ck *self = ckb->get_at<string_to_option_tp_ck>(root_ckb_offset);
        // Second child ckernel is the NA assignment
        self->m_dst_assign_na_offset = ckb_offset - root_ckb_offset;
        nd::callable &assign_na = nd::assign_na::get();
        assign_na.get()->instantiate(assign_na->static_data(), NULL, ckb, dst_tp, dst_arrmeta, nsrc, src_tp,
                                     src_arrmeta, kernreq | kernel_request_data_only, nkwd, kwds, tp_vars);
        ckb_offset = ckb->size();
      }
    };

  } // namespace dynd::nd::detail

  /**
   * A ckernel which assigns option[S] to T.
   */
  struct DYND_API option_to_value_ck : nd::base_strided_kernel<option_to_value_ck, 1> {
    // The default child is the src_is_na ckernel
    size_t m_value_assign_offset;

    ~option_to_value_ck()
    {
      // src_is_na
      get_child()->destroy();
      // value_assign
      get_child(m_value_assign_offset)->destroy();
    }

    void single(char *dst, char *const *src)
    {
      kernel_prefix *src_is_na = get_child();
      kernel_single_t src_is_na_fn = src_is_na->get_function<kernel_single_t>();
      kernel_prefix *value_assign = get_child(m_value_assign_offset);
      kernel_single_t value_assign_fn = value_assign->get_function<kernel_single_t>();
      // Make sure it's not an NA
      bool1 missing = bool1(false);
      src_is_na_fn(src_is_na, reinterpret_cast<char *>(&missing), src);
      if (missing) {
        throw std::overflow_error("cannot assign an NA value to a non-option type");
      }
      // Copy using value assignment
      value_assign_fn(value_assign, dst, src);
    }

    void strided(char *dst, intptr_t dst_stride, char *const *src, const intptr_t *src_stride, size_t count)
    {
      // Two child ckernels
      kernel_prefix *src_is_na = get_child();
      kernel_strided_t src_is_na_fn = src_is_na->get_function<kernel_strided_t>();
      kernel_prefix *value_assign = get_child(m_value_assign_offset);
      kernel_strided_t value_assign_fn = value_assign->get_function<kernel_strided_t>();
      // Process in chunks using the dynd default buffer size
      bool1 missing[DYND_BUFFER_CHUNK_SIZE];
      char *src_copy = src[0];
      while (count > 0) {
        size_t chunk_size = std::min(count, (size_t)DYND_BUFFER_CHUNK_SIZE);
        src_is_na_fn(src_is_na, reinterpret_cast<char *>(missing), 1, &src_copy, src_stride, chunk_size);
        for (size_t i = 0; i < chunk_size; ++i) {
          if (missing[i]) {
            throw std::overflow_error("cannot assign an NA value to a non-option type");
          }
        }
        value_assign_fn(value_assign, dst, dst_stride, &src_copy, src_stride, chunk_size);
        dst += chunk_size * dst_stride;
        src_copy += chunk_size * src_stride[0];
        count -= chunk_size;
      }
    }

    static void instantiate(char *DYND_UNUSED(static_data), char *DYND_UNUSED(data), kernel_builder *ckb,
                            const ndt::type &dst_tp, const char *dst_arrmeta, intptr_t nsrc, const ndt::type *src_tp,
                            const char *const *src_arrmeta, kernel_request_t kernreq, intptr_t nkwd,
                            const nd::array *kwds, const std::map<std::string, ndt::type> &tp_vars)
    {
      intptr_t ckb_offset = ckb->size();
      intptr_t root_ckb_offset = ckb_offset;
      typedef dynd::nd::option_to_value_ck self_type;
      if (dst_tp.get_id() == option_id || src_tp[0].get_id() != option_id) {
        std::stringstream ss;
        ss << "option to value kernel needs value/option types, got " << dst_tp << " and " << src_tp[0];
        throw std::invalid_argument(ss.str());
      }
      const ndt::type &src_val_tp = src_tp[0].extended<ndt::option_type>()->get_value_type();
      ckb->emplace_back<self_type>(kernreq);
      // instantiate src_is_na
      is_na::get()->instantiate(is_na::get()->static_data(), NULL, ckb, ndt::make_type<bool1>(), NULL, nsrc, src_tp,
                                src_arrmeta, kernreq | kernel_request_data_only, 0, nullptr, tp_vars);
      ckb_offset = ckb->size();
      // instantiate value_assign
      ckb->reserve(ckb_offset + sizeof(kernel_prefix));
      self_type *self = ckb->get_at<self_type>(root_ckb_offset);
      self->m_value_assign_offset = ckb_offset - root_ckb_offset;

      assign::get()->instantiate(nd::assign::get()->static_data(), NULL, ckb, dst_tp, dst_arrmeta, 1, &src_val_tp,
                                 src_arrmeta, kernreq | kernel_request_data_only, nkwd, kwds, tp_vars);
    }
  };

  template <int N>
  struct trivial_copy_kernel;

  template <>
  struct trivial_copy_kernel<1> : base_strided_kernel<trivial_copy_kernel<1>, 1> {
    void single(char *dst, char *const *src) { *dst = *src[0]; }
  };

  template <>
  struct trivial_copy_kernel<2> : base_strided_kernel<trivial_copy_kernel<2>, 1> {
    void single(char *dst, char *const *src)
    {
      *reinterpret_cast<int16_t *>(dst) = *reinterpret_cast<int16_t *>(src[0]);
    }
  };

  template <>
  struct trivial_copy_kernel<4> : base_strided_kernel<trivial_copy_kernel<4>, 1> {
    void single(char *dst, char *const *src)
    {
      *reinterpret_cast<int32_t *>(dst) = *reinterpret_cast<int32_t *>(src[0]);
    }
  };

  template <>
  struct trivial_copy_kernel<8> : base_strided_kernel<trivial_copy_kernel<8>, 1> {
    void single(char *dst, char *const *src)
    {
      *reinterpret_cast<int64_t *>(dst) = *reinterpret_cast<int64_t *>(src[0]);
    }
  };

  struct unaligned_copy_ck : base_strided_kernel<unaligned_copy_ck, 1> {
    size_t data_size;

    unaligned_copy_ck(size_t data_size) : data_size(data_size) {}

    void single(char *dst, char *const *src) { memcpy(dst, *src, data_size); }
  };

  namespace detail {

    template <>
    struct assignment_virtual_kernel<fixed_bytes_id, bytes_kind_id, fixed_bytes_id, bytes_kind_id>
        : base_kernel<assignment_virtual_kernel<fixed_bytes_id, bytes_kind_id, fixed_bytes_id, bytes_kind_id>> {
      static void instantiate(char *DYND_UNUSED(static_data), char *DYND_UNUSED(data), kernel_builder *DYND_UNUSED(ckb),
                              const ndt::type &DYND_UNUSED(dst_tp), const char *DYND_UNUSED(dst_arrmeta),
                              intptr_t DYND_UNUSED(nsrc), const ndt::type *DYND_UNUSED(src_tp),
                              const char *const *DYND_UNUSED(src_arrmeta), kernel_request_t DYND_UNUSED(kernreq),
                              intptr_t DYND_UNUSED(nkwd), const nd::array *DYND_UNUSED(kwds),
                              const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars))
      {
        throw std::runtime_error("cannot assign to a fixed_bytes type of a different size");
      }
    };

    template <type_id_t Src0TypeID, assign_error_mode ErrorMode>
    struct assignment_kernel<string_id, string_kind_id, Src0TypeID, int_kind_id, ErrorMode>
        : base_strided_kernel<assignment_kernel<string_id, string_kind_id, Src0TypeID, int_kind_id, ErrorMode>, 1> {
      ndt::type dst_string_tp;
      type_id_t src_id;
      const char *dst_arrmeta;

      assignment_kernel(const ndt::type &dst_string_tp, type_id_t src_id, const char *dst_arrmeta)
          : dst_string_tp(dst_string_tp), src_id(src_id), dst_arrmeta(dst_arrmeta)
      {
      }

      void single(char *dst, char *const *src)
      {
        // TODO: There are much faster ways to do this, but it's very generic!
        //       Also, for floating point values, a printing scheme like
        //       Python's, where it prints the shortest string that's
        //       guaranteed to parse to the same float number, would be
        //       better.
        std::stringstream ss;
        ndt::type(src_id).print_data(ss, NULL, src[0]);
        dst_string_tp->set_from_utf8_string(dst_arrmeta, dst, ss.str(), &eval::default_eval_context);
      }

      static void instantiate(char *DYND_UNUSED(static_data), char *DYND_UNUSED(data), kernel_builder *ckb,
                              const ndt::type &dst_tp, const char *dst_arrmeta, intptr_t DYND_UNUSED(nsrc),
                              const ndt::type *src_tp, const char *const *DYND_UNUSED(src_arrmeta),
                              kernel_request_t kernreq, intptr_t DYND_UNUSED(nkwd), const nd::array *DYND_UNUSED(kwds),
                              const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars))
      {
        ckb->emplace_back<assignment_kernel>(kernreq, dst_tp, src_tp[0].get_id(), dst_arrmeta);
      }
    };

    template <type_id_t Src0TypeID, type_id_t Src0BaseTypeID, assign_error_mode ErrorMode>
    struct assignment_kernel<fixed_string_id, string_kind_id, Src0TypeID, Src0BaseTypeID, ErrorMode>
        : assignment_kernel<string_id, string_kind_id, int32_id, int_kind_id, ErrorMode> {
    };

    template <>
    struct assignment_virtual_kernel<tuple_id, scalar_kind_id, tuple_id, scalar_kind_id>
        : base_kernel<assignment_virtual_kernel<tuple_id, scalar_kind_id, tuple_id, scalar_kind_id>> {
      static void instantiate(char *DYND_UNUSED(static_data), char *DYND_UNUSED(data), kernel_builder *ckb,
                              const ndt::type &dst_tp, const char *dst_arrmeta, intptr_t DYND_UNUSED(nsrc),
                              const ndt::type *src_tp, const char *const *src_arrmeta, kernel_request_t kernreq,
                              intptr_t DYND_UNUSED(nkwd), const nd::array *DYND_UNUSED(kwds),
                              const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars))
      {
        if (dst_tp.extended() == src_tp[0].extended()) {
          make_tuple_identical_assignment_kernel(ckb, dst_tp, dst_arrmeta, src_arrmeta[0], kernreq);
        }
        else if (src_tp[0].get_id() == tuple_id || src_tp[0].get_id() == struct_id) {
          make_tuple_assignment_kernel(ckb, dst_tp, dst_arrmeta, src_tp[0], src_arrmeta[0], kernreq);
        }
        else if (src_tp[0].is_builtin()) {
          make_broadcast_to_tuple_assignment_kernel(ckb, dst_tp, dst_arrmeta, src_tp[0], src_arrmeta[0], kernreq);
        }
        else {
          std::stringstream ss;
          ss << "Cannot assign from " << src_tp[0] << " to " << dst_tp;
          throw dynd::type_error(ss.str());
        }
      }
    };

    template <>
    struct assignment_virtual_kernel<struct_id, tuple_id, struct_id, tuple_id>
        : base_kernel<assignment_virtual_kernel<struct_id, tuple_id, struct_id, tuple_id>> {
      static void instantiate(char *DYND_UNUSED(static_data), char *DYND_UNUSED(data), kernel_builder *ckb,
                              const ndt::type &dst_tp, const char *dst_arrmeta, intptr_t DYND_UNUSED(nsrc),
                              const ndt::type *src_tp, const char *const *src_arrmeta, kernel_request_t kernreq,
                              intptr_t DYND_UNUSED(nkwd), const nd::array *DYND_UNUSED(kwds),
                              const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars))
      {
        if (dst_tp.extended() == src_tp[0].extended()) {
          make_tuple_identical_assignment_kernel(ckb, dst_tp, dst_arrmeta, src_arrmeta[0], kernreq);
          return;
        }
        else if (src_tp[0].get_id() == struct_id) {
          make_struct_assignment_kernel(ckb, dst_tp, dst_arrmeta, src_tp[0], src_arrmeta[0], kernreq);
          return;
        }
        else if (src_tp[0].is_builtin()) {
          make_broadcast_to_tuple_assignment_kernel(ckb, dst_tp, dst_arrmeta, src_tp[0], src_arrmeta[0], kernreq);
          return;
        }

        std::stringstream ss;
        ss << "Cannot assign from " << src_tp[0] << " to " << dst_tp;
        throw dynd::type_error(ss.str());
      }
    };

    template <assign_error_mode ErrorMode>
    struct assignment_kernel<string_id, string_kind_id, fixed_string_id, string_kind_id, ErrorMode>
        : base_strided_kernel<assignment_kernel<string_id, string_kind_id, fixed_string_id, string_kind_id, ErrorMode>,
                              1> {
      string_encoding_t m_dst_encoding, m_src_encoding;
      intptr_t m_src_element_size;
      next_unicode_codepoint_t m_next_fn;
      append_unicode_codepoint_t m_append_fn;

      assignment_kernel(string_encoding_t dst_encoding, string_encoding_t src_encoding, intptr_t src_element_size,
                        next_unicode_codepoint_t next_fn, append_unicode_codepoint_t append_fn)
          : m_dst_encoding(dst_encoding), m_src_encoding(src_encoding), m_src_element_size(src_element_size),
            m_next_fn(next_fn), m_append_fn(append_fn)
      {
      }

      void single(char *dst, char *const *src)
      {
        dynd::string *dst_d = reinterpret_cast<dynd::string *>(dst);
        intptr_t src_charsize = string_encoding_char_size_table[m_src_encoding];
        intptr_t dst_charsize = string_encoding_char_size_table[m_dst_encoding];

        if (dst_d->begin() != NULL) {
          throw std::runtime_error("Cannot assign to an already initialized dynd string");
        }

        char *dst_current;
        const char *src_begin = src[0];
        const char *src_end = src[0] + m_src_element_size;
        next_unicode_codepoint_t next_fn = m_next_fn;
        append_unicode_codepoint_t append_fn = m_append_fn;
        uint32_t cp;

        // Allocate the initial output as the src number of characters + some
        // padding
        // TODO: Don't add padding if the output is not a multi-character encoding
        dynd::string tmp;
        tmp.resize(((src_end - src_begin) / src_charsize + 16) * dst_charsize * 1124 / 1024);
        char *dst_begin = tmp.begin();
        char *dst_end = tmp.end();

        dst_current = dst_begin;
        while (src_begin < src_end) {
          cp = next_fn(src_begin, src_end);
          // Append the codepoint, or increase the allocated memory as necessary
          if (cp != 0) {
            if (dst_end - dst_current >= 8) {
              append_fn(cp, dst_current, dst_end);
            }
            else {
              char *dst_begin_saved = dst_begin;
              tmp.resize(2 * (dst_end - dst_begin));
              dst_begin = tmp.begin();
              dst_end = tmp.end();
              dst_current = dst_begin + (dst_current - dst_begin_saved);

              append_fn(cp, dst_current, dst_end);
            }
          }
          else {
            break;
          }
        }

        // Shrink-wrap the memory to just fit the string
        dst_d->assign(dst_begin, dst_current - dst_begin);
      }

      static void instantiate(char *DYND_UNUSED(static_data), char *DYND_UNUSED(data), kernel_builder *ckb,
                              const ndt::type &dst_tp, const char *DYND_UNUSED(dst_arrmeta), intptr_t DYND_UNUSED(nsrc),
                              const ndt::type *src_tp, const char *const *DYND_UNUSED(src_arrmeta),
                              kernel_request_t kernreq, intptr_t DYND_UNUSED(nkwd), const nd::array *DYND_UNUSED(kwds),
                              const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars))
      {
        ckb->emplace_back<assignment_kernel>(
            kernreq, dst_tp.extended<ndt::base_string_type>()->get_encoding(),
            src_tp[0].extended<ndt::base_string_type>()->get_encoding(), src_tp[0].get_data_size(),
            get_next_unicode_codepoint_function(src_tp[0].extended<ndt::base_string_type>()->get_encoding(), ErrorMode),
            get_append_unicode_codepoint_function(dst_tp.extended<ndt::base_string_type>()->get_encoding(), ErrorMode));
      }
    };

    template <>
    struct assignment_kernel<bool_id, bool_kind_id, string_id, string_kind_id, assign_error_nocheck>
        : base_strided_kernel<assignment_kernel<bool_id, bool_kind_id, string_id, string_kind_id, assign_error_nocheck>,
                              1> {
      ndt::type src_string_tp;
      const char *src_arrmeta;

      assignment_kernel(const ndt::type &src_string_tp, const char *src_arrmeta)
          : src_string_tp(src_string_tp), src_arrmeta(src_arrmeta)
      {
      }

      void single(char *dst, char *const *src)
      {
        // Get the string from the source
        std::string s = reinterpret_cast<const ndt::base_string_type *>(src_string_tp.extended())
                            ->get_utf8_string(src_arrmeta, src[0], assign_error_nocheck);
        trim(s);
        *reinterpret_cast<bool1 *>(dst) = parse<bool>(s.data(), s.data() + s.size(), nocheck);
      }

      static void instantiate(char *DYND_UNUSED(static_data), char *DYND_UNUSED(data), kernel_builder *ckb,
                              const ndt::type &DYND_UNUSED(dst_tp), const char *DYND_UNUSED(dst_arrmeta),
                              intptr_t DYND_UNUSED(nsrc), const ndt::type *src_tp, const char *const *src_arrmeta,
                              kernel_request_t kernreq, intptr_t DYND_UNUSED(nkwd), const nd::array *DYND_UNUSED(kwds),
                              const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars))
      {
        ckb->emplace_back<assignment_kernel>(kernreq, src_tp[0], src_arrmeta[0]);
      }
    };

    template <>
    struct assignment_kernel<bool_id, bool_kind_id, string_id, string_kind_id, assign_error_inexact>
        : base_strided_kernel<assignment_kernel<bool_id, bool_kind_id, string_id, string_kind_id, assign_error_inexact>,
                              1> {
      ndt::type src_string_tp;
      const char *src_arrmeta;

      assignment_kernel(const ndt::type &src_string_tp, const char *src_arrmeta)
          : src_string_tp(src_string_tp), src_arrmeta(src_arrmeta)
      {
      }

      void single(char *dst, char *const *src)
      {
        // Get the string from the source
        std::string s = reinterpret_cast<const ndt::base_string_type *>(src_string_tp.extended())
                            ->get_utf8_string(src_arrmeta, src[0], assign_error_inexact);
        trim(s);
        *reinterpret_cast<bool1 *>(dst) = parse<bool>(s.data(), s.data() + s.size());
      }

      static void instantiate(char *DYND_UNUSED(static_data), char *DYND_UNUSED(data), kernel_builder *ckb,
                              const ndt::type &DYND_UNUSED(dst_tp), const char *DYND_UNUSED(dst_arrmeta),
                              intptr_t DYND_UNUSED(nsrc), const ndt::type *src_tp, const char *const *src_arrmeta,
                              kernel_request_t kernreq, intptr_t DYND_UNUSED(nkwd), const nd::array *DYND_UNUSED(kwds),
                              const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars))
      {
        ckb->emplace_back<assignment_kernel>(kernreq, src_tp[0], src_arrmeta[0]);
      }
    };

    template <>
    struct assignment_kernel<bool_id, bool_kind_id, string_id, string_kind_id, assign_error_default>
        : base_strided_kernel<assignment_kernel<bool_id, bool_kind_id, string_id, string_kind_id, assign_error_default>,
                              1> {
      ndt::type src_string_tp;
      const char *src_arrmeta;

      assignment_kernel(const ndt::type &src_string_tp, const char *src_arrmeta)
          : src_string_tp(src_string_tp), src_arrmeta(src_arrmeta)
      {
      }

      void single(char *dst, char *const *src)
      {
        // Get the string from the source
        std::string s = reinterpret_cast<const ndt::base_string_type *>(src_string_tp.extended())
                            ->get_utf8_string(src_arrmeta, src[0], assign_error_default);
        trim(s);
        *reinterpret_cast<bool1 *>(dst) = parse<bool>(s.data(), s.data() + s.size());
      }

      static void instantiate(char *DYND_UNUSED(static_data), char *DYND_UNUSED(data), kernel_builder *ckb,
                              const ndt::type &DYND_UNUSED(dst_tp), const char *DYND_UNUSED(dst_arrmeta),
                              intptr_t DYND_UNUSED(nsrc), const ndt::type *src_tp, const char *const *src_arrmeta,
                              kernel_request_t kernreq, intptr_t DYND_UNUSED(nkwd), const nd::array *DYND_UNUSED(kwds),
                              const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars))
      {
        ckb->emplace_back<assignment_kernel>(kernreq, src_tp[0], src_arrmeta[0]);
      }
    };

    template <>
    struct assignment_kernel<bool_id, bool_kind_id, string_id, string_kind_id, assign_error_overflow>
        : base_strided_kernel<
              assignment_kernel<bool_id, bool_kind_id, string_id, string_kind_id, assign_error_overflow>, 1> {
      ndt::type src_string_tp;
      const char *src_arrmeta;

      assignment_kernel(const ndt::type &src_string_tp, const char *src_arrmeta)
          : src_string_tp(src_string_tp), src_arrmeta(src_arrmeta)
      {
      }

      void single(char *dst, char *const *src)
      {
        // Get the string from the source
        std::string s = reinterpret_cast<const ndt::base_string_type *>(src_string_tp.extended())
                            ->get_utf8_string(src_arrmeta, src[0], assign_error_overflow);
        trim(s);
        *reinterpret_cast<bool1 *>(dst) = parse<bool>(s.data(), s.data() + s.size());
      }

      static void instantiate(char *DYND_UNUSED(static_data), char *DYND_UNUSED(data), kernel_builder *ckb,
                              const ndt::type &DYND_UNUSED(dst_tp), const char *DYND_UNUSED(dst_arrmeta),
                              intptr_t DYND_UNUSED(nsrc), const ndt::type *src_tp, const char *const *src_arrmeta,
                              kernel_request_t kernreq, intptr_t DYND_UNUSED(nkwd), const nd::array *DYND_UNUSED(kwds),
                              const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars))
      {
        ckb->emplace_back<assignment_kernel>(kernreq, src_tp[0], src_arrmeta[0]);
      }
    };

    template <>
    struct assignment_kernel<bool_id, bool_kind_id, string_id, string_kind_id, assign_error_fractional>
        : base_strided_kernel<
              assignment_kernel<bool_id, bool_kind_id, string_id, string_kind_id, assign_error_fractional>, 1> {
      ndt::type src_string_tp;
      const char *src_arrmeta;

      assignment_kernel(const ndt::type &src_string_tp, const char *src_arrmeta)
          : src_string_tp(src_string_tp), src_arrmeta(src_arrmeta)
      {
      }

      void single(char *dst, char *const *src)
      {
        // Get the string from the source
        std::string s = reinterpret_cast<const ndt::base_string_type *>(src_string_tp.extended())
                            ->get_utf8_string(src_arrmeta, src[0], assign_error_fractional);
        trim(s);
        *reinterpret_cast<bool1 *>(dst) = parse<bool>(s.data(), s.data() + s.size());
      }

      static void instantiate(char *DYND_UNUSED(static_data), char *DYND_UNUSED(data), kernel_builder *ckb,
                              const ndt::type &DYND_UNUSED(dst_tp), const char *DYND_UNUSED(dst_arrmeta),
                              intptr_t DYND_UNUSED(nsrc), const ndt::type *src_tp, const char *const *src_arrmeta,
                              kernel_request_t kernreq, intptr_t DYND_UNUSED(nkwd), const nd::array *DYND_UNUSED(kwds),
                              const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars))
      {
        ckb->emplace_back<assignment_kernel>(kernreq, src_tp[0], src_arrmeta[0]);
      }
    };

    template <type_id_t Src0TypeID, assign_error_mode ErrorMode>
    struct assignment_kernel<Src0TypeID, int_kind_id, string_id, string_kind_id, ErrorMode>
        : base_strided_kernel<assignment_kernel<Src0TypeID, int_kind_id, string_id, string_kind_id, ErrorMode>, 1> {
      typedef typename type_of<Src0TypeID>::type T;

      ndt::type src_string_tp;
      const char *src_arrmeta;

      assignment_kernel(const ndt::type &src_string_tp, const char *src_arrmeta)
          : src_string_tp(src_string_tp), src_arrmeta(src_arrmeta)
      {
      }

      void single(char *dst, char *const *src)
      {
        std::string s = reinterpret_cast<const ndt::base_string_type *>(src_string_tp.extended())
                            ->get_utf8_string(src_arrmeta, src[0], ErrorMode);
        trim(s);
        bool negative = false;
        if (!s.empty() && s[0] == '-') {
          s.erase(0, 1);
          negative = true;
        }
        T result;
        if (ErrorMode == assign_error_nocheck) {
          uint64_t value = parse<uint64_t>(s.data(), s.data() + s.size(), nocheck);
          result = negative ? static_cast<T>(-static_cast<int64_t>(value)) : static_cast<T>(value);
        }
        else {
          bool overflow = false;
          uint64_t value = parse<uint64_t>(s.data(), s.data() + s.size());
          if (overflow || overflow_check<T>::is_overflow(value, negative)) {
            raise_string_cast_overflow_error(ndt::make_type<T>(), src_string_tp, src_arrmeta, src[0]);
          }
          result = negative ? static_cast<T>(-static_cast<int64_t>(value)) : static_cast<T>(value);
        }
        *reinterpret_cast<T *>(dst) = result;
      }

      static void instantiate(char *DYND_UNUSED(static_data), char *DYND_UNUSED(data), kernel_builder *ckb,
                              const ndt::type &DYND_UNUSED(dst_tp), const char *DYND_UNUSED(dst_arrmeta),
                              intptr_t DYND_UNUSED(nsrc), const ndt::type *src_tp, const char *const *src_arrmeta,
                              kernel_request_t kernreq, intptr_t DYND_UNUSED(nkwd), const nd::array *DYND_UNUSED(kwds),
                              const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars))
      {
        ckb->emplace_back<assignment_kernel>(kernreq, src_tp[0], src_arrmeta[0]);
      }
    };

    template <type_id_t DstTypeID>
    struct assignment_virtual_kernel<DstTypeID, uint_kind_id, string_id, string_kind_id>
        : base_strided_kernel<assignment_virtual_kernel<DstTypeID, uint_kind_id, string_id, string_kind_id>, 1> {
      typedef typename type_of<DstTypeID>::type dst_type;

      void single(char *dst, char *const *src)
      {
        *reinterpret_cast<dst_type *>(dst) = parse<dst_type>(*reinterpret_cast<string *>(src[0]));
      }
    };

    template <type_id_t DstTypeID>
    struct assignment_virtual_kernel<DstTypeID, uint_kind_id, string_id, string_kind_id, assign_error_nocheck>
        : base_strided_kernel<
              assignment_virtual_kernel<DstTypeID, uint_kind_id, string_id, string_kind_id, assign_error_nocheck>, 1> {
      typedef typename type_of<DstTypeID>::type dst_type;

      void single(char *dst, char *const *src)
      {
        *reinterpret_cast<dst_type *>(dst) = parse<dst_type>(*reinterpret_cast<string *>(src[0]), nocheck);
      }
    };

    template <assign_error_mode ErrorMode>
    struct assignment_kernel<float16_id, float_kind_id, string_id, string_kind_id, ErrorMode>
        : base_strided_kernel<assignment_kernel<float16_id, float_kind_id, string_id, string_kind_id, ErrorMode>, 1> {
      void single(char *DYND_UNUSED(dst), char *const *DYND_UNUSED(src))
      {
        throw std::runtime_error("TODO: implement string_to_float16_single");
      }
    };

    template <assign_error_mode ErrorMode>
    struct assignment_kernel<float32_id, float_kind_id, string_id, string_kind_id, ErrorMode>
        : base_strided_kernel<assignment_kernel<float32_id, float_kind_id, string_id, string_kind_id, ErrorMode>, 1> {
      ndt::type src_string_tp;
      const char *src_arrmeta;

      assignment_kernel(const ndt::type &src_string_tp, const char *src_arrmeta)
          : src_string_tp(src_string_tp), src_arrmeta(src_arrmeta)
      {
      }

      void single(char *dst, char *const *src)
      {
        // Get the string from the source
        std::string s = reinterpret_cast<const ndt::base_string_type *>(src_string_tp.extended())
                            ->get_utf8_string(src_arrmeta, src[0], ErrorMode);
        trim(s);
        double value = parse<double>(s.data(), s.data() + s.size());
        // Assign double -> float according to the error mode
        char *child_src[1] = {reinterpret_cast<char *>(&value)};
        dynd::nd::detail::assignment_kernel<float32_id, float_kind_id, float64_id, float_kind_id,
                                            ErrorMode>::single_wrapper(NULL, dst, child_src);
      }

      static void instantiate(char *DYND_UNUSED(static_data), char *DYND_UNUSED(data), kernel_builder *ckb,
                              const ndt::type &DYND_UNUSED(dst_tp), const char *DYND_UNUSED(dst_arrmeta),
                              intptr_t DYND_UNUSED(nsrc), const ndt::type *src_tp, const char *const *src_arrmeta,
                              kernel_request_t kernreq, intptr_t DYND_UNUSED(nkwd), const nd::array *DYND_UNUSED(kwds),
                              const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars))
      {
        ckb->emplace_back<assignment_kernel>(kernreq, src_tp[0], src_arrmeta[0]);
      }
    };

    template <assign_error_mode ErrorMode>
    struct assignment_kernel<float64_id, float_kind_id, string_id, string_kind_id, ErrorMode>
        : base_strided_kernel<assignment_kernel<float64_id, float_kind_id, string_id, string_kind_id, ErrorMode>, 1> {
      ndt::type src_string_tp;
      const char *src_arrmeta;

      assignment_kernel(const ndt::type &src_string_tp, const char *src_arrmeta)
          : src_string_tp(src_string_tp), src_arrmeta(src_arrmeta)
      {
      }

      void single(char *dst, char *const *src)
      {
        // Get the string from the source
        std::string s = reinterpret_cast<const ndt::base_string_type *>(src_string_tp.extended())
                            ->get_utf8_string(src_arrmeta, src[0], ErrorMode);
        trim(s);
        double value = parse<double>(s.data(), s.data() + s.size());
        *reinterpret_cast<double *>(dst) = value;
      }

      static void instantiate(char *DYND_UNUSED(static_data), char *DYND_UNUSED(data), kernel_builder *ckb,
                              const ndt::type &DYND_UNUSED(dst_tp), const char *DYND_UNUSED(dst_arrmeta),
                              intptr_t DYND_UNUSED(nsrc), const ndt::type *src_tp, const char *const *src_arrmeta,
                              kernel_request_t kernreq, intptr_t DYND_UNUSED(nkwd), const nd::array *DYND_UNUSED(kwds),
                              const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars))
      {
        ckb->emplace_back<assignment_kernel>(kernreq, src_tp[0], src_arrmeta[0]);
      }
    };

    template <assign_error_mode ErrorMode>
    struct assignment_kernel<float128_id, float_kind_id, string_id, string_kind_id, ErrorMode>
        : base_strided_kernel<assignment_kernel<float128_id, float_kind_id, string_id, string_kind_id, ErrorMode>, 1> {
      void single(char *DYND_UNUSED(dst), char *const *DYND_UNUSED(src))
      {
        throw std::runtime_error("TODO: implement string_to_float128_single");
      }
    };

    template <assign_error_mode ErrorMode>
    struct assignment_kernel<complex_float32_id, complex_kind_id, string_id, string_kind_id, ErrorMode>
        : base_strided_kernel<
              assignment_kernel<complex_float32_id, complex_kind_id, string_id, string_kind_id, ErrorMode>, 1> {
      void single(char *DYND_UNUSED(dst), char *const *DYND_UNUSED(src))
      {
        throw std::runtime_error("TODO: implement string_to_complex_float32_single");
      }
    };

    template <assign_error_mode ErrorMode>
    struct assignment_kernel<complex_float64_id, complex_kind_id, string_id, string_kind_id, ErrorMode>
        : base_strided_kernel<
              assignment_kernel<complex_float64_id, complex_kind_id, string_id, string_kind_id, ErrorMode>, 1> {
      void single(char *DYND_UNUSED(dst), char *const *DYND_UNUSED(src))
      {
        throw std::runtime_error("TODO: implement string_to_complex_float64_single");
      }
    };

    template <type_id_t DstTypeID, assign_error_mode ErrorMode>
    struct assignment_kernel<DstTypeID, int_kind_id, fixed_string_id, string_kind_id, ErrorMode>
        : assignment_kernel<DstTypeID, int_kind_id, string_id, string_kind_id, ErrorMode> {
    };

    template <assign_error_mode ErrorMode>
    struct assignment_kernel<fixed_string_id, string_kind_id, fixed_string_id, string_kind_id, ErrorMode>
        : base_strided_kernel<
              assignment_kernel<fixed_string_id, string_kind_id, fixed_string_id, string_kind_id, ErrorMode>, 1> {
      next_unicode_codepoint_t m_next_fn;
      append_unicode_codepoint_t m_append_fn;
      intptr_t m_dst_data_size, m_src_data_size;
      bool m_overflow_check;

      assignment_kernel(next_unicode_codepoint_t next_fn, append_unicode_codepoint_t append_fn, intptr_t dst_data_size,
                        intptr_t src_data_size, bool overflow_check)
          : m_next_fn(next_fn), m_append_fn(append_fn), m_dst_data_size(dst_data_size), m_src_data_size(src_data_size),
            m_overflow_check(overflow_check)
      {
      }

      void single(char *dst, char *const *src)
      {
        char *dst_end = dst + m_dst_data_size;
        const char *src_end = src[0] + m_src_data_size;
        next_unicode_codepoint_t next_fn = m_next_fn;
        append_unicode_codepoint_t append_fn = m_append_fn;
        uint32_t cp = 0;

        char *src_copy = src[0];
        while (src_copy < src_end && dst < dst_end) {
          cp = next_fn(const_cast<const char *&>(src_copy), src_end);
          // The fixed_string type uses null-terminated strings
          if (cp == 0) {
            // Null-terminate the destination string, and we're done
            memset(dst, 0, dst_end - dst);
            return;
          }
          else {
            append_fn(cp, dst, dst_end);
          }
        }
        if (src_copy < src_end) {
          if (m_overflow_check) {
            throw std::runtime_error("Input string is too large to convert to "
                                     "destination fixed-size string");
          }
        }
        else if (dst < dst_end) {
          memset(dst, 0, dst_end - dst);
        }
      }

      static void instantiate(char *DYND_UNUSED(static_data), char *DYND_UNUSED(data), kernel_builder *ckb,
                              const ndt::type &dst_tp, const char *DYND_UNUSED(dst_arrmeta), intptr_t DYND_UNUSED(nsrc),
                              const ndt::type *src_tp, const char *const *DYND_UNUSED(src_arrmeta),
                              kernel_request_t kernreq, intptr_t DYND_UNUSED(nkwd), const nd::array *DYND_UNUSED(kwds),
                              const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars))
      {
        const ndt::fixed_string_type *src_fs = src_tp[0].extended<ndt::fixed_string_type>();
        ckb->emplace_back<assignment_kernel>(
            kernreq, get_next_unicode_codepoint_function(src_fs->get_encoding(), ErrorMode),
            get_append_unicode_codepoint_function(dst_tp.extended<ndt::fixed_string_type>()->get_encoding(), ErrorMode),
            dst_tp.get_data_size(), src_fs->get_data_size(), ErrorMode != assign_error_nocheck);
      }
    };

    template <assign_error_mode ErrorMode>
    struct assignment_kernel<char_id, string_kind_id, char_id, string_kind_id, ErrorMode>
        : assignment_kernel<fixed_string_id, string_kind_id, fixed_string_id, string_kind_id, ErrorMode> {
    };

    template <assign_error_mode ErrorMode>
    struct assignment_kernel<char_id, string_kind_id, fixed_string_id, string_kind_id, ErrorMode>
        : assignment_kernel<fixed_string_id, string_kind_id, fixed_string_id, string_kind_id, ErrorMode> {
    };

    struct adapt_assign_to_kernel : base_kernel<adapt_assign_to_kernel> {
      static void instantiate(char *DYND_UNUSED(static_data), char *data, kernel_builder *ckb, const ndt::type &dst_tp,
                              const char *dst_arrmeta, intptr_t nsrc, const ndt::type *DYND_UNUSED(src_tp),
                              const char *const *src_arrmeta, kernel_request_t kernreq, intptr_t nkwd,
                              const nd::array *kwds, const std::map<std::string, ndt::type> &tp_vars)
      {
        const callable &inverse = dst_tp.extended<ndt::adapt_type>()->get_inverse();
        const ndt::type &value_tp = dst_tp.value_type();
        inverse->instantiate(inverse->static_data(), data, ckb, dst_tp.storage_type(), dst_arrmeta, nsrc, &value_tp,
                             src_arrmeta, kernreq, nkwd, kwds, tp_vars);
      }
    };

    struct adapt_assign_from_kernel : base_strided_kernel<adapt_assign_from_kernel, 1> {
      intptr_t forward_offset;
      array buffer;

      adapt_assign_from_kernel(const ndt::type &buffer_tp) : buffer(empty(buffer_tp)) {}

      ~adapt_assign_from_kernel()
      {
        get_child()->destroy();
        get_child(forward_offset)->destroy();
      }

      void single(char *dst, char *const *src)
      {
        get_child()->single(buffer.data(), src);

        char *child_src[1] = {buffer.data()};
        get_child(forward_offset)->single(dst, child_src);
      }

      static void instantiate(char *DYND_UNUSED(static_data), char *data, kernel_builder *ckb, const ndt::type &dst_tp,
                              const char *dst_arrmeta, intptr_t nsrc, const ndt::type *src_tp,
                              const char *const *src_arrmeta, kernel_request_t kernreq, intptr_t nkwd,
                              const nd::array *kwds, const std::map<std::string, ndt::type> &tp_vars)
      {
        intptr_t ckb_offset = ckb->size();
        const ndt::type &storage_tp = src_tp[0].storage_type();
        if (storage_tp.is_expression()) {
          const callable &forward = src_tp[0].extended<ndt::adapt_type>()->get_forward();

          intptr_t self_offset = ckb_offset;
          ckb->emplace_back<adapt_assign_from_kernel>(kernreq, storage_tp.get_canonical_type());
          ckb_offset = ckb->size();

          nd::assign::get()->instantiate(nd::assign::get()->static_data(), data, ckb, storage_tp.get_canonical_type(),
                                         dst_arrmeta, nsrc, &storage_tp, src_arrmeta, kernel_request_single, nkwd, kwds,
                                         tp_vars);
          ckb_offset = ckb->size();
          intptr_t forward_offset = ckb_offset - self_offset;
          ndt::type src_tp2[1] = {storage_tp.get_canonical_type()};
          forward->instantiate(forward->static_data(), data, ckb, dst_tp, dst_arrmeta, nsrc, src_tp2, src_arrmeta,
                               kernel_request_single, nkwd, kwds, tp_vars);
          ckb_offset = ckb->size();
          ckb->get_at<adapt_assign_from_kernel>(self_offset)->forward_offset = forward_offset;
        }
        else {
          const callable &forward = src_tp[0].extended<ndt::adapt_type>()->get_forward();

          ndt::type src_tp2[1] = {storage_tp.get_canonical_type()};
          forward->instantiate(forward->static_data(), data, ckb, dst_tp, dst_arrmeta, nsrc, src_tp2, src_arrmeta,
                               kernreq, nkwd, kwds, tp_vars);
          ckb_offset = ckb->size();
        }
      }
    };

    template <assign_error_mode ErrorMode>
    struct assignment_kernel<fixed_string_id, string_kind_id, string_id, string_kind_id, ErrorMode>
        : base_strided_kernel<assignment_kernel<fixed_string_id, string_kind_id, string_id, string_kind_id, ErrorMode>,
                              1> {
      next_unicode_codepoint_t m_next_fn;
      append_unicode_codepoint_t m_append_fn;
      intptr_t m_dst_data_size;
      bool m_overflow_check;

      assignment_kernel(next_unicode_codepoint_t next_fn, append_unicode_codepoint_t append_fn, intptr_t dst_data_size,
                        bool overflow_check)
          : m_next_fn(next_fn), m_append_fn(append_fn), m_dst_data_size(dst_data_size), m_overflow_check(overflow_check)
      {
      }

      void single(char *dst, char *const *src)
      {
        char *dst_end = dst + m_dst_data_size;
        const dynd::string *src_d = reinterpret_cast<const dynd::string *>(src[0]);
        const char *src_begin = src_d->begin();
        const char *src_end = src_d->end();
        next_unicode_codepoint_t next_fn = m_next_fn;
        append_unicode_codepoint_t append_fn = m_append_fn;
        uint32_t cp;

        while (src_begin < src_end && dst < dst_end) {
          cp = next_fn(src_begin, src_end);
          append_fn(cp, dst, dst_end);
        }
        if (src_begin < src_end) {
          if (m_overflow_check) {
            throw std::runtime_error("Input string is too large to "
                                     "convert to destination "
                                     "fixed-size string");
          }
        }
        else if (dst < dst_end) {
          memset(dst, 0, dst_end - dst);
        }
      }

      static void instantiate(char *DYND_UNUSED(static_data), char *DYND_UNUSED(data), kernel_builder *ckb,
                              const ndt::type &dst_tp, const char *DYND_UNUSED(dst_arrmeta), intptr_t DYND_UNUSED(nsrc),
                              const ndt::type *src_tp, const char *const *DYND_UNUSED(src_arrmeta),
                              kernel_request_t kernreq, intptr_t DYND_UNUSED(nkwd), const nd::array *DYND_UNUSED(kwds),
                              const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars))
      {
        const ndt::base_string_type *src_fs = src_tp[0].extended<ndt::base_string_type>();
        ckb->emplace_back<assignment_kernel>(
            kernreq, get_next_unicode_codepoint_function(src_fs->get_encoding(), ErrorMode),
            get_append_unicode_codepoint_function(dst_tp.extended<ndt::fixed_string_type>()->get_encoding(), ErrorMode),
            dst_tp.get_data_size(), ErrorMode != assign_error_nocheck);
      }
    };

    template <assign_error_mode ErrorMode>
    struct assignment_kernel<char_id, string_kind_id, string_id, string_kind_id, ErrorMode>
        : base_kernel<assignment_kernel<char_id, string_kind_id, string_id, string_kind_id, ErrorMode>> {
      static void instantiate(char *DYND_UNUSED(static_data), char *DYND_UNUSED(data), kernel_builder *ckb,
                              const ndt::type &dst_tp, const char *DYND_UNUSED(dst_arrmeta), intptr_t DYND_UNUSED(nsrc),
                              const ndt::type *src_tp, const char *const *DYND_UNUSED(src_arrmeta),
                              kernel_request_t kernreq, intptr_t DYND_UNUSED(nkwd), const nd::array *DYND_UNUSED(kwds),
                              const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars))
      {
        const ndt::base_string_type *src_fs = src_tp[0].extended<ndt::base_string_type>();
        ckb->emplace_back<assignment_kernel<fixed_string_id, string_kind_id, string_id, string_kind_id, ErrorMode>>(
            kernreq, get_next_unicode_codepoint_function(src_fs->get_encoding(), ErrorMode),
            get_append_unicode_codepoint_function(dst_tp.extended<ndt::char_type>()->get_encoding(), ErrorMode),
            dst_tp.get_data_size(), ErrorMode != assign_error_nocheck);
      }
    };

    template <assign_error_mode ErrorMode>
    struct assignment_kernel<fixed_string_id, string_kind_id, char_id, string_kind_id, ErrorMode>
        : assignment_kernel<fixed_string_id, string_kind_id, fixed_string_id, string_kind_id, ErrorMode> {
    };

    template <assign_error_mode ErrorMode>
    struct assignment_kernel<string_id, string_kind_id, char_id, string_kind_id, ErrorMode>
        : base_kernel<assignment_kernel<string_id, string_kind_id, char_id, string_kind_id, ErrorMode>> {
      static void instantiate(char *DYND_UNUSED(static_data), char *DYND_UNUSED(data), kernel_builder *ckb,
                              const ndt::type &dst_tp, const char *DYND_UNUSED(dst_arrmeta), intptr_t DYND_UNUSED(nsrc),
                              const ndt::type *src_tp, const char *const *DYND_UNUSED(src_arrmeta),
                              kernel_request_t kernreq, intptr_t DYND_UNUSED(nkwd), const nd::array *DYND_UNUSED(kwds),
                              const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars))
      {
        ckb->emplace_back<assignment_kernel<string_id, string_kind_id, fixed_string_id, string_kind_id, ErrorMode>>(
            kernreq, dst_tp.extended<ndt::base_string_type>()->get_encoding(),
            src_tp[0].extended<ndt::char_type>()->get_encoding(), src_tp[0].get_data_size(),
            get_next_unicode_codepoint_function(src_tp[0].extended<ndt::char_type>()->get_encoding(), ErrorMode),
            get_append_unicode_codepoint_function(dst_tp.extended<ndt::base_string_type>()->get_encoding(), ErrorMode));
      }
    };

    struct assignment_option_kernel : base_kernel<assignment_option_kernel> {
      static void instantiate(char *DYND_UNUSED(static_data), char *DYND_UNUSED(data), kernel_builder *ckb,
                              const ndt::type &dst_tp, const char *dst_arrmeta, intptr_t DYND_UNUSED(nsrc),
                              const ndt::type *src_tp, const char *const *src_arrmeta, kernel_request_t kernreq,
                              intptr_t nkwd, const nd::array *kwds, const std::map<std::string, ndt::type> &tp_vars)
      {
        ndt::type val_dst_tp =
            dst_tp.get_id() == option_id ? dst_tp.extended<ndt::option_type>()->get_value_type() : dst_tp;
        ndt::type val_src_tp =
            src_tp[0].get_id() == option_id ? src_tp[0].extended<ndt::option_type>()->get_value_type() : src_tp[0];
        assign::get()->instantiate(nd::assign::get()->static_data(), NULL, ckb, val_dst_tp, dst_arrmeta, 1, &val_src_tp,
                                   src_arrmeta, kernreq, nkwd, kwds, tp_vars);
      }
    };

    template <>
    struct assignment_virtual_kernel<pointer_id, any_kind_id, pointer_id, any_kind_id>
        : base_strided_kernel<assignment_virtual_kernel<pointer_id, any_kind_id, pointer_id, any_kind_id>, 1> {
      ~assignment_virtual_kernel() { get_child()->destroy(); }

      void single(char *dst, char *const *src)
      {
        kernel_prefix *copy_value = get_child();
        kernel_single_t copy_value_fn = copy_value->get_function<kernel_single_t>();
        // The src value is a pointer, and copy_value_fn expects a pointer
        // to that pointer
        char **src_ptr = reinterpret_cast<char **>(src[0]);
        copy_value_fn(copy_value, dst, src_ptr);
      }

      static void instantiate(char *DYND_UNUSED(static_data), char *DYND_UNUSED(data), kernel_builder *ckb,
                              const ndt::type &dst_tp, const char *dst_arrmeta, intptr_t DYND_UNUSED(nsrc),
                              const ndt::type *src_tp, const char *const *src_arrmeta, kernel_request_t kernreq,
                              intptr_t nkwd, const nd::array *kwds, const std::map<std::string, ndt::type> &tp_vars)
      {
        ckb->emplace_back<assignment_virtual_kernel>(kernreq);

        const char *child_src_arrmeta = src_arrmeta[0] + sizeof(pointer_type_arrmeta);
        assign::get()->instantiate(nd::assign::get()->static_data(), NULL, ckb,
                                   dst_tp.extended<ndt::pointer_type>()->get_target_type(), dst_arrmeta, 1,
                                   &src_tp[0].extended<ndt::pointer_type>()->get_target_type(), &child_src_arrmeta,
                                   kernel_request_single, nkwd, kwds, tp_vars);
      }
    };

    template <>
    struct assignment_virtual_kernel<type_id, scalar_kind_id, type_id, scalar_kind_id>
        : base_strided_kernel<assignment_virtual_kernel<type_id, scalar_kind_id, type_id, scalar_kind_id>, 1> {
      void single(char *dst, char *const *src)
      {
        *reinterpret_cast<ndt::type *>(dst) = *reinterpret_cast<ndt::type *>(src[0]);
      }
    };

    template <assign_error_mode ErrorMode>
    struct assignment_kernel<type_id, scalar_kind_id, string_id, string_kind_id, ErrorMode>
        : base_strided_kernel<assignment_kernel<type_id, scalar_kind_id, string_id, string_kind_id, ErrorMode>, 1> {
      ndt::type src_string_dt;
      const char *src_arrmeta;

      assignment_kernel(const ndt::type &src0_tp, const char *src0_arrmeta)
          : src_string_dt(src0_tp), src_arrmeta(src0_arrmeta)
      {
      }

      void single(char *dst, char *const *src)
      {
        const std::string &s =
            src_string_dt.extended<ndt::base_string_type>()->get_utf8_string(src_arrmeta, src[0], ErrorMode);
        ndt::type(s).swap(*reinterpret_cast<ndt::type *>(dst));
      }

      static void instantiate(char *DYND_UNUSED(static_data), char *DYND_UNUSED(data), kernel_builder *ckb,
                              const ndt::type &DYND_UNUSED(dst_tp), const char *DYND_UNUSED(dst_arrmeta),
                              intptr_t DYND_UNUSED(nsrc), const ndt::type *src_tp, const char *const *src_arrmeta,
                              kernel_request_t kernreq, intptr_t DYND_UNUSED(nkwd), const nd::array *DYND_UNUSED(kwds),
                              const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars))
      {
        ckb->emplace_back<assignment_kernel>(kernreq, src_tp[0], src_arrmeta[0]);
      }
    };

    template <assign_error_mode ErrorMode>
    struct assignment_kernel<string_id, string_kind_id, type_id, scalar_kind_id, ErrorMode>
        : base_strided_kernel<assignment_kernel<string_id, string_kind_id, type_id, scalar_kind_id, ErrorMode>, 1> {
      ndt::type dst_string_dt;
      const char *dst_arrmeta;

      assignment_kernel(const ndt::type &dst_tp, const char *dst_arrmeta)
          : dst_string_dt(dst_tp), dst_arrmeta(dst_arrmeta)
      {
      }

      void single(char *dst, char *const *src)
      {
        std::stringstream ss;
        ss << *reinterpret_cast<ndt::type *>(src[0]);
        dst_string_dt->set_from_utf8_string(dst_arrmeta, dst, ss.str(), &eval::default_eval_context);
      }

      static void instantiate(char *DYND_UNUSED(static_data), char *DYND_UNUSED(data), kernel_builder *ckb,
                              const ndt::type &dst_tp, const char *dst_arrmeta, intptr_t DYND_UNUSED(nsrc),
                              const ndt::type *DYND_UNUSED(src_tp), const char *const *DYND_UNUSED(src_arrmeta),
                              kernel_request_t kernreq, intptr_t DYND_UNUSED(nkwd), const nd::array *DYND_UNUSED(kwds),
                              const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars))
      {
        ckb->emplace_back<assignment_kernel>(kernreq, dst_tp, dst_arrmeta);
      }
    };

  } // namespace dynd::nd::detail

  template <type_id_t DstTypeID, type_id_t Src0TypeID>
  struct assignment_kernel : detail::assignment_virtual_kernel<DstTypeID, base_id_of<DstTypeID>::value, Src0TypeID,
                                                               base_id_of<Src0TypeID>::value> {
  };

} // namespace dynd::nd

namespace ndt {

  template <type_id_t DstTypeID, type_id_t Src0TypeID>
  struct traits<nd::assignment_kernel<DstTypeID, Src0TypeID>> {
    static type equivalent()
    {
      return callable_type::make(type(DstTypeID), {type(Src0TypeID)}, {"error_mode"},
                                 {ndt::make_type<ndt::option_type>(ndt::make_type<assign_error_mode>())});
    }
  };

} // namespace dynd::ndt

} // namespace dynd
