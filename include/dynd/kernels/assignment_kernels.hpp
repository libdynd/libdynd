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
#include <dynd/kernels/expression_assignment_kernels.hpp>
#include <dynd/kernels/cuda_launch.hpp>
#include <dynd/kernels/tuple_assignment_kernels.hpp>
#include <dynd/kernels/struct_assignment_kernels.hpp>
#include <dynd/kernels/base_kernel.hpp>
#include <dynd/kernels/base_virtual_kernel.hpp>
#include <dynd/kernels/option_assignment_kernels.hpp>
#include <dynd/kernels/pointer_assignment_kernels.hpp>
#include <dynd/eval/eval_context.hpp>
#include <dynd/typed_data_assign.hpp>
#include <dynd/types/type_id.hpp>
#include <dynd/types/datetime_type.hpp>
#include <dynd/types/date_type.hpp>
#include <dynd/types/categorical_type.hpp>
#include <dynd/types/char_type.hpp>
#include <dynd/types/new_adapt_type.hpp>
#include <dynd/types/time_type.hpp>
#include <dynd/types/fixed_bytes_type.hpp>
#include <dynd/types/option_type.hpp>
#include <dynd/types/fixed_string_type.hpp>
#include <dynd/types/property_type.hpp>
#include <dynd/parser_util.hpp>
#include <map>

#if defined(_MSC_VER)
// Tell the visual studio compiler we're accessing the FPU flags
#pragma fenv_access(on)
#endif

namespace dynd {

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

template <class T>
struct overflow_check;
template <>
struct overflow_check<int8_t> {
  inline static bool is_overflow(uint64_t value, bool negative)
  {
    return (value & ~0x7fULL) != 0 && !(negative && value == 0x80ULL);
  }
};
template <>
struct overflow_check<int16_t> {
  inline static bool is_overflow(uint64_t value, bool negative)
  {
    return (value & ~0x7fffULL) != 0 && !(negative && value == 0x8000ULL);
  }
};
template <>
struct overflow_check<int32_t> {
  inline static bool is_overflow(uint64_t value, bool negative)
  {
    return (value & ~0x7fffffffULL) != 0 && !(negative && value == 0x80000000ULL);
  }
};
template <>
struct overflow_check<int64_t> {
  inline static bool is_overflow(uint64_t value, bool negative)
  {
    return (value & ~0x7fffffffffffffffULL) != 0 && !(negative && value == 0x8000000000000000ULL);
  }
};
template <>
struct overflow_check<int128> {
  inline static bool is_overflow(uint128 value, bool negative)
  {
    return (value.m_hi & ~0x7fffffffffffffffULL) != 0 &&
           !(negative && value.m_hi == 0x8000000000000000ULL && value.m_lo == 0ULL);
  }
};
template <>
struct overflow_check<uint8_t> {
  inline static bool is_overflow(uint64_t value) { return (value & ~0xffULL) != 0; }
};
template <>
struct overflow_check<uint16_t> {
  inline static bool is_overflow(uint64_t value) { return (value & ~0xffffULL) != 0; }
};
template <>
struct overflow_check<uint32_t> {
  inline static bool is_overflow(uint64_t value) { return (value & ~0xffffffffULL) != 0; }
};
template <>
struct overflow_check<uint64_t> {
  inline static bool is_overflow(uint64_t DYND_UNUSED(value)) { return false; }
};

inline void raise_string_cast_error(const ndt::type &dst_tp, const ndt::type &string_tp, const char *arrmeta,
                                    const char *data)
{
  std::stringstream ss;
  ss << "cannot cast string ";
  string_tp.print_data(ss, arrmeta, data);
  ss << " to " << dst_tp;
  throw std::invalid_argument(ss.str());
}

inline void raise_string_cast_overflow_error(const ndt::type &dst_tp, const ndt::type &string_tp, const char *arrmeta,
                                             const char *data)
{
  std::stringstream ss;
  ss << "overflow converting string ";
  string_tp.print_data(ss, arrmeta, data);
  ss << " to " << dst_tp;
  throw std::overflow_error(ss.str());
}

template <typename DstType, typename SrcType>
typename std::enable_if<(sizeof(DstType) < sizeof(SrcType)) && is_signed<DstType>::value && is_signed<SrcType>::value,
                        bool>::type
is_overflow(SrcType src)
{
  return src < static_cast<SrcType>(std::numeric_limits<DstType>::min()) ||
         src > static_cast<SrcType>(std::numeric_limits<DstType>::max());
}

template <typename DstType, typename SrcType>
typename std::enable_if<(sizeof(DstType) >= sizeof(SrcType)) && is_signed<DstType>::value && is_signed<SrcType>::value,
                        bool>::type
is_overflow(SrcType DYND_UNUSED(src))
{
  return false;
}

template <typename DstType, typename SrcType>
typename std::enable_if<(sizeof(DstType) < sizeof(SrcType)) && is_signed<DstType>::value && is_unsigned<SrcType>::value,
                        bool>::type
is_overflow(SrcType src)
{
  return src > static_cast<SrcType>(std::numeric_limits<DstType>::max());
}

template <typename DstType, typename SrcType>
typename std::enable_if<
    (sizeof(DstType) >= sizeof(SrcType)) && is_signed<DstType>::value && is_unsigned<SrcType>::value, bool>::type
is_overflow(SrcType DYND_UNUSED(src))
{
  return false;
}

template <typename DstType, typename SrcType>
typename std::enable_if<(sizeof(DstType) < sizeof(SrcType)) && is_unsigned<DstType>::value && is_signed<SrcType>::value,
                        bool>::type
is_overflow(SrcType src)
{
  return src < static_cast<SrcType>(0) || static_cast<SrcType>(std::numeric_limits<DstType>::max()) < src;
}

template <typename DstType, typename SrcType>
typename std::enable_if<
    (sizeof(DstType) >= sizeof(SrcType)) && is_unsigned<DstType>::value && is_signed<SrcType>::value, bool>::type
is_overflow(SrcType src)
{
  return src < static_cast<SrcType>(0);
}

template <typename DstType, typename SrcType>
typename std::enable_if<
    (sizeof(DstType) < sizeof(SrcType)) && is_unsigned<DstType>::value && is_unsigned<SrcType>::value, bool>::type
is_overflow(SrcType src)
{
  return static_cast<SrcType>(std::numeric_limits<DstType>::max()) < src;
}

template <typename DstType, typename SrcType>
typename std::enable_if<
    (sizeof(DstType) >= sizeof(SrcType)) && is_unsigned<DstType>::value && is_unsigned<SrcType>::value, bool>::type
is_overflow(SrcType DYND_UNUSED(src))
{
  return false;
}

namespace nd {

  template <typename UIntType>
  struct category_to_categorical_kernel_extra : base_kernel<category_to_categorical_kernel_extra<UIntType>, 1> {
    typedef category_to_categorical_kernel_extra self_type;

    ndt::type dst_cat_tp;
    const char *src_arrmeta;

    // Assign from an input matching the category type to a categorical type
    void single(char *dst, char *const *src)
    {
      uint32_t src_val = dst_cat_tp.extended<ndt::categorical_type>()->get_value_from_category(src_arrmeta, src[0]);
      *reinterpret_cast<UIntType *>(dst) = src_val;
    }

    static void destruct(ckernel_prefix *self)
    {
      self_type *e = reinterpret_cast<self_type *>(self);
      e->dst_cat_tp.~type();
    }
  };

  // Assign from a categorical type to some other type
  template <typename UIntType>
  struct categorical_to_other_kernel : base_kernel<categorical_to_other_kernel<UIntType>, 1> {
    typedef categorical_to_other_kernel extra_type;

    ndt::type src_cat_tp;

    void single(char *dst, char *const *src)
    {
      ckernel_prefix *echild = this->get_child();
      expr_single_t opchild = echild->get_function<expr_single_t>();

      uint32_t value = *reinterpret_cast<const UIntType *>(src[0]);
      char *src_val = const_cast<char *>(
          reinterpret_cast<const ndt::categorical_type *>(src_cat_tp.extended())->get_category_data_from_value(value));
      opchild(echild, dst, &src_val);
    }

    static void destruct(ckernel_prefix *self)
    {
      extra_type *e = reinterpret_cast<extra_type *>(self);
      e->src_cat_tp.~type();
      self->get_child(sizeof(extra_type))->destroy();
    }
  };

  namespace detail {

    template <type_id_t DstTypeID, type_kind_t DstTypeKind, type_id_t Src0TypeID, type_kind_t Src0TypeKind,
              assign_error_mode ErrorMode>
    struct assignment_kernel;

    template <type_id_t DstTypeID, type_kind_t DstTypeKind, type_id_t Src0TypeID, type_kind_t Src0TypeKind>
    struct assignment_virtual_kernel
        : base_virtual_kernel<assignment_virtual_kernel<DstTypeID, DstTypeKind, Src0TypeID, Src0TypeKind>> {
      static intptr_t instantiate(char *static_data, char *data, void *ckb, intptr_t ckb_offset,
                                  const ndt::type &dst_tp, const char *dst_arrmeta, intptr_t nsrc,
                                  const ndt::type *src_tp, const char *const *src_arrmeta, kernel_request_t kernreq,
                                  const eval::eval_context *ectx, intptr_t nkwd, const nd::array *kwds,
                                  const std::map<std::string, ndt::type> &tp_vars)
      {
        switch (ectx->errmode) {
        case assign_error_nocheck:
          return assignment_kernel<DstTypeID, DstTypeKind, Src0TypeID, Src0TypeKind, assign_error_nocheck>::instantiate(
              static_data, data, ckb, ckb_offset, dst_tp, dst_arrmeta, nsrc, src_tp, src_arrmeta, kernreq, ectx, nkwd,
              kwds, tp_vars);
        case assign_error_overflow:
          return assignment_kernel<DstTypeID, DstTypeKind, Src0TypeID, Src0TypeKind,
                                   assign_error_overflow>::instantiate(static_data, data, ckb, ckb_offset, dst_tp,
                                                                       dst_arrmeta, nsrc, src_tp, src_arrmeta, kernreq,
                                                                       ectx, nkwd, kwds, tp_vars);
        case assign_error_fractional:
          return assignment_kernel<DstTypeID, DstTypeKind, Src0TypeID, Src0TypeKind,
                                   assign_error_fractional>::instantiate(static_data, data, ckb, ckb_offset, dst_tp,
                                                                         dst_arrmeta, nsrc, src_tp, src_arrmeta,
                                                                         kernreq, ectx, nkwd, kwds, tp_vars);
        case assign_error_inexact:
          return assignment_kernel<DstTypeID, DstTypeKind, Src0TypeID, Src0TypeKind, assign_error_inexact>::instantiate(
              static_data, data, ckb, ckb_offset, dst_tp, dst_arrmeta, nsrc, src_tp, src_arrmeta, kernreq, ectx, nkwd,
              kwds, tp_vars);
        default:
          throw std::runtime_error("error");
        }
      }
    };

    template <type_id_t DstTypeID, type_kind_t DstTypeKind, type_id_t Src0TypeID, type_kind_t Src0TypeKind,
              assign_error_mode ErrorMode>
    struct assignment_kernel
        : base_kernel<assignment_kernel<DstTypeID, DstTypeKind, Src0TypeID, Src0TypeKind, ErrorMode>, 1> {
      typedef typename type_of<DstTypeID>::type dst_type;
      typedef typename type_of<Src0TypeID>::type src_type;

      void single(char *dst, char *const *src)
      {
        DYND_TRACE_ASSIGNMENT(static_cast<dst_type>(*reinterpret_cast<src_type *>(src[0])), dst_type,
                              *reinterpret_cast<src_type *>(src[0]), src_type);

        *reinterpret_cast<dst_type *>(dst) = static_cast<dst_type>(*reinterpret_cast<src_type *>(src[0]));
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
          ss << "assignment from " << ndt::type::make<src0_type>() << " to "
             << ndt::type::make<dst_type>();
          ss << "with error mode " << assign_error_nocheck << " is not
    implemented";
          throw std::runtime_error(ss.str());
    #endif
        }
      };
    */

    // Complex floating point -> non-complex with no error checking
    template <type_id_t DstTypeID, type_id_t Src0TypeID>
    struct assignment_kernel<DstTypeID, sint_kind, Src0TypeID, complex_kind, assign_error_nocheck>
        : base_kernel<assignment_kernel<DstTypeID, sint_kind, Src0TypeID, complex_kind, assign_error_nocheck>, 1> {
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
    struct assignment_kernel<DstTypeID, uint_kind, Src0TypeID, complex_kind, assign_error_nocheck>
        : base_kernel<assignment_kernel<DstTypeID, uint_kind, Src0TypeID, complex_kind, assign_error_nocheck>, 1> {
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
    struct assignment_kernel<DstTypeID, real_kind, Src0TypeID, complex_kind, assign_error_nocheck>
        : base_kernel<assignment_kernel<DstTypeID, real_kind, Src0TypeID, complex_kind, assign_error_nocheck>, 1> {
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
    struct assignment_kernel<DstTypeID, complex_kind, Src0TypeID, sint_kind, assign_error_nocheck>
        : base_kernel<assignment_kernel<DstTypeID, complex_kind, Src0TypeID, sint_kind, assign_error_nocheck>, 1> {
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
    struct assignment_kernel<DstTypeID, complex_kind, Src0TypeID, sint_kind, assign_error_inexact>
        : base_kernel<assignment_kernel<DstTypeID, complex_kind, Src0TypeID, sint_kind, assign_error_inexact>, 1> {
      typedef typename type_of<DstTypeID>::type dst_type;
      typedef typename type_of<Src0TypeID>::type src0_type;

      void single(char *dst, char *const *src)
      {
        src0_type s = *reinterpret_cast<src0_type *>(src[0]);
        typename dst_type::value_type d = static_cast<typename dst_type::value_type>(s);

        DYND_TRACE_ASSIGNMENT(d, dst_type, s, src0_type);

        if (static_cast<src0_type>(d) != s) {
          std::stringstream ss;
          ss << "inexact value while assigning " << ndt::type::make<src0_type>() << " value ";
          ss << s << " to " << ndt::type::make<dst_type>() << " value " << d;
          throw std::runtime_error(ss.str());
        }
        *reinterpret_cast<dst_type *>(dst) = d;
      }
    };

    // Signed int -> complex floating point with other checking
    template <type_id_t DstTypeID, type_id_t Src0TypeID>
    struct assignment_kernel<DstTypeID, complex_kind, Src0TypeID, sint_kind, assign_error_overflow>
        : assignment_kernel<DstTypeID, complex_kind, Src0TypeID, sint_kind, assign_error_nocheck> {
    };

    template <type_id_t DstTypeID, type_id_t Src0TypeID>
    struct assignment_kernel<DstTypeID, complex_kind, Src0TypeID, sint_kind, assign_error_fractional>
        : assignment_kernel<DstTypeID, complex_kind, Src0TypeID, sint_kind, assign_error_nocheck> {
    };

    // Anything -> boolean with no checking
    template <type_id_t Src0TypeID, type_kind_t Src0TypeKind>
    struct assignment_kernel<bool_type_id, bool_kind, Src0TypeID, Src0TypeKind, assign_error_nocheck>
        : base_kernel<assignment_kernel<bool_type_id, bool_kind, Src0TypeID, Src0TypeKind, assign_error_nocheck>, 1> {
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
    struct assignment_kernel<DstTypeID, real_kind, Src0TypeID, uint_kind, assign_error_inexact>
        : base_kernel<assignment_kernel<DstTypeID, real_kind, Src0TypeID, uint_kind, assign_error_inexact>, 1> {
      typedef typename type_of<DstTypeID>::type dst_type;
      typedef typename type_of<Src0TypeID>::type src0_type;

      void single(char *dst, char *const *src)
      {
        src0_type s = *reinterpret_cast<src0_type *>(src[0]);
        dst_type d = static_cast<dst_type>(s);

        DYND_TRACE_ASSIGNMENT(d, dst_type, s, src0_type);

        if (static_cast<src0_type>(d) != s) {
          std::stringstream ss;
          ss << "inexact value while assigning " << ndt::type::make<src0_type>() << " value ";
          ss << s << " to " << ndt::type::make<dst_type>() << " value " << d;
          throw std::runtime_error(ss.str());
        }
        *reinterpret_cast<dst_type *>(dst) = d;
      }
    };

    // Unsigned int -> floating point with other checking
    template <type_id_t DstTypeID, type_id_t Src0TypeID>
    struct assignment_kernel<DstTypeID, real_kind, Src0TypeID, uint_kind, assign_error_overflow>
        : assignment_kernel<DstTypeID, real_kind, Src0TypeID, uint_kind, assign_error_nocheck> {
    };

    template <type_id_t DstTypeID, type_id_t Src0TypeID>
    struct assignment_kernel<DstTypeID, real_kind, Src0TypeID, uint_kind, assign_error_fractional>
        : assignment_kernel<DstTypeID, real_kind, Src0TypeID, uint_kind, assign_error_nocheck> {
    };

    // Unsigned int -> complex floating point with no checking
    template <type_id_t DstTypeID, type_id_t Src0TypeID>
    struct assignment_kernel<DstTypeID, complex_kind, Src0TypeID, uint_kind, assign_error_nocheck>
        : base_kernel<assignment_kernel<DstTypeID, complex_kind, Src0TypeID, uint_kind, assign_error_nocheck>, 1> {
      typedef typename type_of<DstTypeID>::type dst_type;
      typedef typename type_of<Src0TypeID>::type src0_type;

      DYND_CUDA_HOST_DEVICE void single(char *dst, char *const *src)
      {
        src0_type s = *reinterpret_cast<src0_type *>(src[0]);

        DYND_TRACE_ASSIGNMENT(d, dst_type, s, src0_type);

        *reinterpret_cast<dst_type *>(dst) = static_cast<typename dst_type::value_type>(s);
      }
    };

    // Unsigned int -> complex floating point with inexact checking
    template <type_id_t DstTypeID, type_id_t Src0TypeID>
    struct assignment_kernel<DstTypeID, complex_kind, Src0TypeID, uint_kind, assign_error_inexact>
        : base_kernel<assignment_kernel<DstTypeID, complex_kind, Src0TypeID, uint_kind, assign_error_inexact>, 1> {
      typedef typename type_of<DstTypeID>::type dst_type;
      typedef typename type_of<Src0TypeID>::type src0_type;

      void single(char *dst, char *const *src)
      {
        src0_type s = *reinterpret_cast<src0_type *>(src[0]);
        typename dst_type::value_type d = static_cast<typename dst_type::value_type>(s);

        DYND_TRACE_ASSIGNMENT(d, dst_type, s, src0_type);

        if (static_cast<src0_type>(d) != s) {
          std::stringstream ss;
          ss << "inexact value while assigning " << ndt::type::make<src0_type>() << " value ";
          ss << s << " to " << ndt::type::make<dst_type>() << " value " << d;
          throw std::runtime_error(ss.str());
        }
        *reinterpret_cast<dst_type *>(dst) = d;
      }
    };

    // Unsigned int -> complex floating point with other checking
    template <type_id_t DstTypeID, type_id_t Src0TypeID>
    struct assignment_kernel<DstTypeID, complex_kind, Src0TypeID, uint_kind, assign_error_overflow>
        : assignment_kernel<DstTypeID, complex_kind, Src0TypeID, uint_kind, assign_error_nocheck> {
    };

    template <type_id_t DstTypeID, type_id_t Src0TypeID>
    struct assignment_kernel<DstTypeID, complex_kind, Src0TypeID, uint_kind, assign_error_fractional>
        : assignment_kernel<DstTypeID, complex_kind, Src0TypeID, uint_kind, assign_error_nocheck> {
    };

    // Floating point -> signed int with overflow checking
    template <type_id_t DstTypeID, type_id_t Src0TypeID>
    struct assignment_kernel<DstTypeID, sint_kind, Src0TypeID, real_kind, assign_error_overflow>
        : base_kernel<assignment_kernel<DstTypeID, sint_kind, Src0TypeID, real_kind, assign_error_overflow>, 1> {
      typedef typename type_of<DstTypeID>::type dst_type;
      typedef typename type_of<Src0TypeID>::type src0_type;

      void single(char *dst, char *const *src)
      {
        src0_type s = *reinterpret_cast<src0_type *>(src[0]);

        DYND_TRACE_ASSIGNMENT(static_cast<dst_type>(s), dst_type, s, src_type);

        if (s < std::numeric_limits<dst_type>::min() || std::numeric_limits<dst_type>::max() < s) {
          std::stringstream ss;
          ss << "overflow while assigning " << ndt::type::make<src0_type>() << " value ";
          ss << s << " to " << ndt::type::make<dst_type>();
          throw std::overflow_error(ss.str());
        }
        *reinterpret_cast<dst_type *>(dst) = static_cast<dst_type>(s);
      }
    };

    // Floating point -> signed int with fractional checking
    template <type_id_t DstTypeID, type_id_t Src0TypeID>
    struct assignment_kernel<DstTypeID, sint_kind, Src0TypeID, real_kind, assign_error_fractional>
        : base_kernel<assignment_kernel<DstTypeID, sint_kind, Src0TypeID, real_kind, assign_error_fractional>, 1> {
      typedef typename type_of<DstTypeID>::type dst_type;
      typedef typename type_of<Src0TypeID>::type src0_type;

      void single(char *dst, char *const *src)
      {
        src0_type s = *reinterpret_cast<src0_type *>(src[0]);

        DYND_TRACE_ASSIGNMENT(static_cast<dst_type>(s), dst_type, s, src0_type);

        if (s < std::numeric_limits<dst_type>::min() || std::numeric_limits<dst_type>::max() < s) {
          std::stringstream ss;
          ss << "overflow while assigning " << ndt::type::make<src0_type>() << " value ";
          ss << s << " to " << ndt::type::make<dst_type>();
          throw std::overflow_error(ss.str());
        }

        if (floor(s) != s) {
          std::stringstream ss;
          ss << "fractional part lost while assigning " << ndt::type::make<src0_type>() << " value ";
          ss << s << " to " << ndt::type::make<dst_type>();
          throw std::runtime_error(ss.str());
        }
        *reinterpret_cast<dst_type *>(dst) = static_cast<dst_type>(s);
      }
    };

    // Floating point -> signed int with other checking
    template <type_id_t DstTypeID, type_id_t Src0TypeID>
    struct assignment_kernel<DstTypeID, sint_kind, Src0TypeID, real_kind, assign_error_inexact>
        : assignment_kernel<DstTypeID, sint_kind, Src0TypeID, real_kind, assign_error_fractional> {
    };

    // Complex floating point -> signed int with overflow checking
    template <type_id_t DstTypeID, type_id_t Src0TypeID>
    struct assignment_kernel<DstTypeID, sint_kind, Src0TypeID, complex_kind, assign_error_overflow>
        : base_kernel<assignment_kernel<DstTypeID, sint_kind, Src0TypeID, complex_kind, assign_error_overflow>, 1> {
      typedef typename type_of<DstTypeID>::type dst_type;
      typedef typename type_of<Src0TypeID>::type src0_type;

      void single(char *dst, char *const *src)
      {
        src0_type s = *reinterpret_cast<src0_type *>(src[0]);

        DYND_TRACE_ASSIGNMENT(static_cast<dst_type>(s.real()), dst_type, s, src0_type);

        if (s.imag() != 0) {
          std::stringstream ss;
          ss << "loss of imaginary component while assigning " << ndt::type::make<src0_type>() << " value ";
          ss << s << " to " << ndt::type::make<dst_type>();
          throw std::runtime_error(ss.str());
        }

        if (s.real() < std::numeric_limits<dst_type>::min() || std::numeric_limits<dst_type>::max() < s.real()) {
          std::stringstream ss;
          ss << "overflow while assigning " << ndt::type::make<src0_type>() << " value ";
          ss << s << " to " << ndt::type::make<dst_type>();
          throw std::overflow_error(ss.str());
        }
        *reinterpret_cast<dst_type *>(dst) = static_cast<dst_type>(s.real());
      }
    };

    // Complex floating point -> signed int with fractional checking
    template <type_id_t DstTypeID, type_id_t Src0TypeID>
    struct assignment_kernel<DstTypeID, sint_kind, Src0TypeID, complex_kind, assign_error_fractional>
        : base_kernel<assignment_kernel<DstTypeID, sint_kind, Src0TypeID, complex_kind, assign_error_fractional>, 1> {
      typedef typename type_of<DstTypeID>::type dst_type;
      typedef typename type_of<Src0TypeID>::type src0_type;

      void single(char *dst, char *const *src)
      {
        src0_type s = *reinterpret_cast<src0_type *>(src[0]);

        DYND_TRACE_ASSIGNMENT(static_cast<dst_type>(s.real()), dst_type, s, src0_type);

        if (s.imag() != 0) {
          std::stringstream ss;
          ss << "loss of imaginary component while assigning " << ndt::type::make<src0_type>() << " value ";
          ss << s << " to " << ndt::type::make<dst_type>();
          throw std::runtime_error(ss.str());
        }

        if (s.real() < std::numeric_limits<dst_type>::min() || std::numeric_limits<dst_type>::max() < s.real()) {
          std::stringstream ss;
          ss << "overflow while assigning " << ndt::type::make<src0_type>() << " value ";
          ss << s << " to " << ndt::type::make<dst_type>();
          throw std::overflow_error(ss.str());
        }

        if (std::floor(s.real()) != s.real()) {
          std::stringstream ss;
          ss << "fractional part lost while assigning " << ndt::type::make<src0_type>() << " value ";
          ss << s << " to " << ndt::type::make<dst_type>();
          throw std::runtime_error(ss.str());
        }
        *reinterpret_cast<dst_type *>(dst) = static_cast<dst_type>(s.real());
      }
    };

    // Complex floating point -> signed int with other checking
    template <type_id_t DstTypeID, type_id_t Src0TypeID>
    struct assignment_kernel<DstTypeID, sint_kind, Src0TypeID, complex_kind, assign_error_inexact>
        : assignment_kernel<DstTypeID, sint_kind, Src0TypeID, complex_kind, assign_error_fractional> {
    };

    // Floating point -> unsigned int with overflow checking
    template <type_id_t DstTypeID, type_id_t Src0TypeID>
    struct assignment_kernel<DstTypeID, uint_kind, Src0TypeID, real_kind, assign_error_overflow>
        : base_kernel<assignment_kernel<DstTypeID, uint_kind, Src0TypeID, real_kind, assign_error_overflow>, 1> {
      typedef typename type_of<DstTypeID>::type dst_type;
      typedef typename type_of<Src0TypeID>::type src0_type;

      void single(char *dst, char *const *src)
      {
        src0_type s = *reinterpret_cast<src0_type *>(src[0]);

        DYND_TRACE_ASSIGNMENT(static_cast<dst_type>(s), dst_type, s, src0_type);

        if (s < 0 || std::numeric_limits<dst_type>::max() < s) {
          std::stringstream ss;
          ss << "overflow while assigning " << ndt::type::make<src0_type>() << " value ";
          ss << s << " to " << ndt::type::make<dst_type>();
          throw std::overflow_error(ss.str());
        }
        *reinterpret_cast<dst_type *>(dst) = static_cast<dst_type>(s);
      }
    };

    // Floating point -> unsigned int with fractional checking
    template <type_id_t DstTypeID, type_id_t Src0TypeID>
    struct assignment_kernel<DstTypeID, uint_kind, Src0TypeID, real_kind, assign_error_fractional>
        : base_kernel<assignment_kernel<DstTypeID, uint_kind, Src0TypeID, real_kind, assign_error_fractional>, 1> {
      typedef typename type_of<DstTypeID>::type dst_type;
      typedef typename type_of<Src0TypeID>::type src0_type;

      void single(char *dst, char *const *src)
      {
        src0_type s = *reinterpret_cast<src0_type *>(src[0]);

        DYND_TRACE_ASSIGNMENT(static_cast<dst_type>(s), dst_type, s, src0_type);

        if (s < 0 || std::numeric_limits<dst_type>::max() < s) {
          std::stringstream ss;
          ss << "overflow while assigning " << ndt::type::make<src0_type>() << " value ";
          ss << s << " to " << ndt::type::make<dst_type>();
          throw std::overflow_error(ss.str());
        }

        if (floor(s) != s) {
          std::stringstream ss;
          ss << "fractional part lost while assigning " << ndt::type::make<src0_type>() << " value ";
          ss << s << " to " << ndt::type::make<dst_type>();
          throw std::runtime_error(ss.str());
        }
        *reinterpret_cast<dst_type *>(dst) = static_cast<dst_type>(s);
      }
    };

    // Floating point -> unsigned int with other checking
    template <type_id_t DstTypeID, type_id_t Src0TypeID>
    struct assignment_kernel<DstTypeID, uint_kind, Src0TypeID, real_kind, assign_error_inexact>
        : assignment_kernel<DstTypeID, uint_kind, Src0TypeID, real_kind, assign_error_fractional> {
    };

    // Complex floating point -> unsigned int with overflow checking
    template <type_id_t DstTypeID, type_id_t Src0TypeID>
    struct assignment_kernel<DstTypeID, uint_kind, Src0TypeID, complex_kind, assign_error_overflow>
        : base_kernel<assignment_kernel<DstTypeID, uint_kind, Src0TypeID, complex_kind, assign_error_overflow>, 1> {
      typedef typename type_of<DstTypeID>::type dst_type;
      typedef typename type_of<Src0TypeID>::type src0_type;

      void single(char *dst, char *const *src)
      {
        src0_type s = *reinterpret_cast<src0_type *>(src[0]);

        DYND_TRACE_ASSIGNMENT(static_cast<dst_type>(s.real()), dst_type, s, complex<src_real_type>);

        if (s.imag() != 0) {
          std::stringstream ss;
          ss << "loss of imaginary component while assigning " << ndt::type::make<src0_type>() << " value ";
          ss << s << " to " << ndt::type::make<dst_type>();
          throw std::runtime_error(ss.str());
        }

        if (s.real() < 0 || std::numeric_limits<dst_type>::max() < s.real()) {
          std::stringstream ss;
          ss << "overflow while assigning " << ndt::type::make<src0_type>() << " value ";
          ss << s << " to " << ndt::type::make<dst_type>();
          throw std::overflow_error(ss.str());
        }
        *reinterpret_cast<dst_type *>(dst) = static_cast<dst_type>(s.real());
      }
    };

    // Complex floating point -> unsigned int with fractional checking
    template <type_id_t DstTypeID, type_id_t Src0TypeID>
    struct assignment_kernel<DstTypeID, uint_kind, Src0TypeID, complex_kind, assign_error_fractional>
        : base_kernel<assignment_kernel<DstTypeID, uint_kind, Src0TypeID, complex_kind, assign_error_fractional>, 1> {
      typedef typename type_of<DstTypeID>::type dst_type;
      typedef typename type_of<Src0TypeID>::type src0_type;

      void single(char *dst, char *const *src)
      {
        src0_type s = *reinterpret_cast<src0_type *>(src[0]);

        DYND_TRACE_ASSIGNMENT(static_cast<dst_type>(s.real()), dst_type, s, src0_type);

        if (s.imag() != 0) {
          std::stringstream ss;
          ss << "loss of imaginary component while assigning " << ndt::type::make<src0_type>() << " value ";
          ss << s << " to " << ndt::type::make<dst_type>();
          throw std::runtime_error(ss.str());
        }

        if (s.real() < 0 || std::numeric_limits<dst_type>::max() < s.real()) {
          std::stringstream ss;
          ss << "overflow while assigning " << ndt::type::make<src0_type>() << " value ";
          ss << s << " to " << ndt::type::make<dst_type>();
          throw std::overflow_error(ss.str());
        }

        if (std::floor(s.real()) != s.real()) {
          std::stringstream ss;
          ss << "fractional part lost while assigning " << ndt::type::make<src0_type>() << " value ";
          ss << s << " to " << ndt::type::make<dst_type>();
          throw std::runtime_error(ss.str());
        }
        *reinterpret_cast<dst_type *>(dst) = static_cast<dst_type>(s.real());
      }
    };

    // Complex floating point -> unsigned int with other checking
    template <type_id_t DstTypeID, type_id_t Src0TypeID>
    struct assignment_kernel<DstTypeID, uint_kind, Src0TypeID, complex_kind, assign_error_inexact>
        : assignment_kernel<DstTypeID, uint_kind, Src0TypeID, complex_kind, assign_error_fractional> {
    };

    // float -> float with no checking
    template <>
    struct assignment_kernel<float32_type_id, real_kind, float32_type_id, real_kind, assign_error_overflow>
        : assignment_kernel<float32_type_id, real_kind, float32_type_id, real_kind, assign_error_nocheck> {
    };

    template <>
    struct assignment_kernel<float32_type_id, real_kind, float32_type_id, real_kind, assign_error_fractional>
        : assignment_kernel<float32_type_id, real_kind, float32_type_id, real_kind, assign_error_nocheck> {
    };

    template <>
    struct assignment_kernel<float32_type_id, real_kind, float32_type_id, real_kind, assign_error_inexact>
        : assignment_kernel<float32_type_id, real_kind, float32_type_id, real_kind, assign_error_nocheck> {
    };

    // complex<float> -> complex<float> with no checking
    template <>
    struct assignment_kernel<complex_float32_type_id, complex_kind, complex_float32_type_id, complex_kind,
                             assign_error_overflow>
        : assignment_kernel<complex_float32_type_id, complex_kind, complex_float32_type_id, complex_kind,
                            assign_error_nocheck> {
    };

    template <>
    struct assignment_kernel<complex_float32_type_id, complex_kind, complex_float32_type_id, complex_kind,
                             assign_error_fractional>
        : assignment_kernel<complex_float32_type_id, complex_kind, complex_float32_type_id, complex_kind,
                            assign_error_nocheck> {
    };

    template <>
    struct assignment_kernel<complex_float32_type_id, complex_kind, complex_float32_type_id, complex_kind,
                             assign_error_inexact>
        : assignment_kernel<complex_float32_type_id, complex_kind, complex_float32_type_id, complex_kind,
                            assign_error_nocheck> {
    };

    // float -> double with no checking
    template <>
    struct assignment_kernel<float64_type_id, real_kind, float32_type_id, real_kind, assign_error_overflow>
        : assignment_kernel<float64_type_id, real_kind, float32_type_id, real_kind, assign_error_nocheck> {
    };

    template <>
    struct assignment_kernel<float64_type_id, real_kind, float32_type_id, real_kind, assign_error_fractional>
        : assignment_kernel<float64_type_id, real_kind, float32_type_id, real_kind, assign_error_nocheck> {
    };

    template <>
    struct assignment_kernel<float64_type_id, real_kind, float32_type_id, real_kind, assign_error_inexact>
        : assignment_kernel<float64_type_id, real_kind, float32_type_id, real_kind, assign_error_nocheck> {
    };

    // complex<float> -> complex<double> with no checking
    template <>
    struct assignment_kernel<complex_float64_type_id, complex_kind, complex_float32_type_id, complex_kind,
                             assign_error_overflow>
        : assignment_kernel<complex_float64_type_id, complex_kind, complex_float32_type_id, complex_kind,
                            assign_error_nocheck> {
    };

    template <>
    struct assignment_kernel<complex_float64_type_id, complex_kind, complex_float32_type_id, complex_kind,
                             assign_error_fractional>
        : assignment_kernel<complex_float64_type_id, complex_kind, complex_float32_type_id, complex_kind,
                            assign_error_nocheck> {
    };

    template <>
    struct assignment_kernel<complex_float64_type_id, complex_kind, complex_float32_type_id, complex_kind,
                             assign_error_inexact>
        : assignment_kernel<complex_float64_type_id, complex_kind, complex_float32_type_id, complex_kind,
                            assign_error_nocheck> {
    };

    // double -> double with no checking
    template <>
    struct assignment_kernel<float64_type_id, real_kind, float64_type_id, real_kind, assign_error_overflow>
        : assignment_kernel<float64_type_id, real_kind, float64_type_id, real_kind, assign_error_nocheck> {
    };

    template <>
    struct assignment_kernel<float64_type_id, real_kind, float64_type_id, real_kind, assign_error_fractional>
        : assignment_kernel<float64_type_id, real_kind, float64_type_id, real_kind, assign_error_nocheck> {
    };

    template <>
    struct assignment_kernel<float64_type_id, real_kind, float64_type_id, real_kind, assign_error_inexact>
        : assignment_kernel<float64_type_id, real_kind, float64_type_id, real_kind, assign_error_nocheck> {
    };

    // complex<double> -> complex<double> with no checking
    template <>
    struct assignment_kernel<complex_float64_type_id, complex_kind, complex_float64_type_id, complex_kind,
                             assign_error_overflow>
        : assignment_kernel<complex_float64_type_id, complex_kind, complex_float64_type_id, complex_kind,
                            assign_error_nocheck> {
    };

    template <>
    struct assignment_kernel<complex_float64_type_id, complex_kind, complex_float64_type_id, complex_kind,
                             assign_error_fractional>
        : assignment_kernel<complex_float64_type_id, complex_kind, complex_float64_type_id, complex_kind,
                            assign_error_nocheck> {
    };

    template <>
    struct assignment_kernel<complex_float64_type_id, complex_kind, complex_float64_type_id, complex_kind,
                             assign_error_inexact>
        : assignment_kernel<complex_float64_type_id, complex_kind, complex_float64_type_id, complex_kind,
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
            ss << "overflow while assigning " << ndt::type::make<src0_type>()
               << " value ";
            ss << s << " to " << ndt::type::make<dst_type>();
            throw std::overflow_error(ss.str());
          }
    #else
          src0_type sd = s;
          if (isfinite(sd) && (sd < -std::numeric_limits<dst_type>::max() ||
                               sd > std::numeric_limits<dst_type>::max())) {
            std::stringstream ss;
            ss << "overflow while assigning " << ndt::type::make<src0_type>()
               << " value ";
            ss << s << " to " << ndt::type::make<dst_type>();
            throw std::overflow_error(ss.str());
          }
          *reinterpret_cast<dst_type *>(dst) = static_cast<dst_type>(sd);
    #endif // DYND_USE_FPSTATUS
        }
      };
    */

    // real -> real with overflow checking
    template <type_id_t DstTypeID, type_id_t Src0TypeID>
    struct assignment_kernel<DstTypeID, real_kind, Src0TypeID, real_kind, assign_error_overflow>
        : base_kernel<assignment_kernel<DstTypeID, real_kind, Src0TypeID, real_kind, assign_error_overflow>, 1> {
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
          ss << "overflow while assigning " << ndt::type::make<src0_type>() << " value ";
          ss << s << " to " << ndt::type::make<dst_type>();
          throw std::overflow_error(ss.str());
        }
#else
        src0_type sd = s;
        if (isfinite(sd) && (sd < -std::numeric_limits<dst_type>::max() || sd > std::numeric_limits<dst_type>::max())) {
          std::stringstream ss;
          ss << "overflow while assigning " << ndt::type::make<src0_type>() << " value ";
          ss << s << " to " << ndt::type::make<dst_type>();
          throw std::overflow_error(ss.str());
        }
        *reinterpret_cast<dst_type *>(dst) = static_cast<dst_type>(sd);
#endif // DYND_USE_FPSTATUS
      }
    };

    // real -> real with fractional checking
    template <type_id_t DstTypeID, type_id_t Src0TypeID>
    struct assignment_kernel<DstTypeID, real_kind, Src0TypeID, real_kind, assign_error_fractional>
        : assignment_kernel<DstTypeID, real_kind, Src0TypeID, real_kind, assign_error_overflow> {
    };

    // real -> real with inexact checking
    template <type_id_t DstTypeID, type_id_t Src0TypeID>
    struct assignment_kernel<DstTypeID, real_kind, Src0TypeID, real_kind, assign_error_inexact>
        : base_kernel<assignment_kernel<DstTypeID, real_kind, Src0TypeID, real_kind, assign_error_inexact>, 1> {
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
          ss << "overflow while assigning " << ndt::type::make<src0_type>() << " value ";
          ss << s << " to " << ndt::type::make<dst_type>();
          throw std::overflow_error(ss.str());
        }
#else
        if (isfinite(s) && (s < -std::numeric_limits<dst_type>::max() || s > std::numeric_limits<dst_type>::max())) {
          std::stringstream ss;
          ss << "overflow while assigning " << ndt::type::make<src0_type>() << " value ";
          ss << s << " to " << ndt::type::make<dst_type>();
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
          ss << "inexact precision loss while assigning " << ndt::type::make<src0_type>() << " value ";
          ss << s << " to " << ndt::type::make<dst_type>();
          throw std::runtime_error(ss.str());
        }
        *reinterpret_cast<dst_type *>(dst) = d;
      }
    };

    // Anything -> boolean with overflow checking
    template <type_id_t Src0TypeID, type_kind_t Src0TypeKind>
    struct assignment_kernel<bool_type_id, bool_kind, Src0TypeID, Src0TypeKind, assign_error_overflow>
        : base_kernel<assignment_kernel<bool_type_id, bool_kind, Src0TypeID, Src0TypeKind, assign_error_overflow>, 1> {
      typedef typename type_of<Src0TypeID>::type src0_type;

      void single(char *dst, char *const *src)
      {
        src0_type s = *reinterpret_cast<src0_type *>(src[0]);

        DYND_TRACE_ASSIGNMENT((bool)(s != src0_type(0)), bool1, s, src0_type);

        if (s == src0_type(0)) {
          *dst = false;
        }
        else if (s == src0_type(1)) {
          *dst = true;
        }
        else {
          std::stringstream ss;
          ss << "overflow while assigning " << ndt::type::make<src0_type>() << " value ";
          ss << s << " to " << ndt::type::make<bool1>();
          throw std::overflow_error(ss.str());
        }
      }
    };

    // Anything -> boolean with other error checking
    template <type_id_t Src0TypeID, type_kind_t Src0TypeKind>
    struct assignment_kernel<bool_type_id, bool_kind, Src0TypeID, Src0TypeKind, assign_error_fractional>
        : assignment_kernel<bool_type_id, bool_kind, Src0TypeID, Src0TypeKind, assign_error_overflow> {
    };

    template <type_id_t Src0TypeID, type_kind_t Src0TypeKind>
    struct assignment_kernel<bool_type_id, bool_kind, Src0TypeID, Src0TypeKind, assign_error_inexact>
        : assignment_kernel<bool_type_id, bool_kind, Src0TypeID, Src0TypeKind, assign_error_overflow> {
    };

    // Boolean -> boolean with other error checking
    template <>
    struct assignment_kernel<bool_type_id, bool_kind, bool_type_id, bool_kind, assign_error_overflow>
        : assignment_kernel<bool_type_id, bool_kind, bool_type_id, bool_kind, assign_error_nocheck> {
    };

    template <>
    struct assignment_kernel<bool_type_id, bool_kind, bool_type_id, bool_kind, assign_error_fractional>
        : assignment_kernel<bool_type_id, bool_kind, bool_type_id, bool_kind, assign_error_nocheck> {
    };

    template <>
    struct assignment_kernel<bool_type_id, bool_kind, bool_type_id, bool_kind, assign_error_inexact>
        : assignment_kernel<bool_type_id, bool_kind, bool_type_id, bool_kind, assign_error_nocheck> {
    };

    // Boolean -> anything with other error checking
    template <type_id_t DstTypeID, type_kind_t DstTypeKind>
    struct assignment_kernel<DstTypeID, DstTypeKind, bool_type_id, bool_kind, assign_error_overflow>
        : assignment_kernel<DstTypeID, DstTypeKind, bool_type_id, bool_kind, assign_error_nocheck> {
    };

    template <type_id_t DstTypeID, type_kind_t DstTypeKind>
    struct assignment_kernel<DstTypeID, DstTypeKind, bool_type_id, bool_kind, assign_error_fractional>
        : assignment_kernel<DstTypeID, DstTypeKind, bool_type_id, bool_kind, assign_error_nocheck> {
    };

    template <type_id_t DstTypeID, type_kind_t DstTypeKind>
    struct assignment_kernel<DstTypeID, DstTypeKind, bool_type_id, bool_kind, assign_error_inexact>
        : assignment_kernel<DstTypeID, DstTypeKind, bool_type_id, bool_kind, assign_error_nocheck> {
    };

    // Signed int -> signed int with overflow checking
    template <type_id_t DstTypeID, type_id_t Src0TypeID>
    struct assignment_kernel<DstTypeID, sint_kind, Src0TypeID, sint_kind, assign_error_overflow>
        : base_kernel<assignment_kernel<DstTypeID, sint_kind, Src0TypeID, sint_kind, assign_error_overflow>, 1> {
      typedef typename type_of<DstTypeID>::type dst_type;
      typedef typename type_of<Src0TypeID>::type src0_type;

      void single(char *dst, char *const *src)
      {
        src0_type s = *reinterpret_cast<src0_type *>(src[0]);

        if (is_overflow<dst_type>(s)) {
          std::stringstream ss;
          ss << "overflow while assigning " << ndt::type::make<src0_type>() << " value ";
          ss << s << " to " << ndt::type::make<dst_type>();
          throw std::overflow_error(ss.str());
        }
        *reinterpret_cast<dst_type *>(dst) = static_cast<dst_type>(s);
      }
    };

    // Signed int -> signed int with other error checking
    template <type_id_t DstTypeID, type_id_t Src0TypeID>
    struct assignment_kernel<DstTypeID, sint_kind, Src0TypeID, sint_kind, assign_error_fractional>
        : assignment_kernel<DstTypeID, sint_kind, Src0TypeID, sint_kind, assign_error_overflow> {
    };

    template <type_id_t DstTypeID, type_id_t Src0TypeID>
    struct assignment_kernel<DstTypeID, sint_kind, Src0TypeID, sint_kind, assign_error_inexact>
        : assignment_kernel<DstTypeID, sint_kind, Src0TypeID, sint_kind, assign_error_overflow> {
    };

    // Unsigned int -> signed int with overflow checking just when sizeof(dst)
    // <=
    // sizeof(src)
    template <type_id_t DstTypeID, type_id_t Src0TypeID>
    struct assignment_kernel<DstTypeID, sint_kind, Src0TypeID, uint_kind, assign_error_overflow>
        : base_kernel<assignment_kernel<DstTypeID, sint_kind, Src0TypeID, uint_kind, assign_error_overflow>, 1> {
      typedef typename type_of<DstTypeID>::type dst_type;
      typedef typename type_of<Src0TypeID>::type src0_type;

      void single(char *dst, char *const *src)
      {
        src0_type s = *reinterpret_cast<src0_type *>(src[0]);

        DYND_TRACE_ASSIGNMENT(static_cast<dst_type>(s), dst_type, s, src0_type);

        if (is_overflow<dst_type>(s)) {
          std::stringstream ss;
          ss << "overflow while assigning " << ndt::type::make<src0_type>() << " value ";
          ss << s << " to " << ndt::type::make<dst_type>();
          throw std::overflow_error(ss.str());
        }
        *reinterpret_cast<dst_type *>(dst) = static_cast<dst_type>(s);
      }
    };

    // Unsigned int -> signed int with other error checking
    template <type_id_t DstTypeID, type_id_t Src0TypeID>
    struct assignment_kernel<DstTypeID, sint_kind, Src0TypeID, uint_kind, assign_error_fractional>
        : assignment_kernel<DstTypeID, sint_kind, Src0TypeID, uint_kind, assign_error_overflow> {
    };

    template <type_id_t DstTypeID, type_id_t Src0TypeID>
    struct assignment_kernel<DstTypeID, sint_kind, Src0TypeID, uint_kind, assign_error_inexact>
        : assignment_kernel<DstTypeID, sint_kind, Src0TypeID, uint_kind, assign_error_overflow> {
    };

    // Signed int -> unsigned int with positive overflow checking just when
    // sizeof(dst) < sizeof(src)
    template <type_id_t DstTypeID, type_id_t Src0TypeID>
    struct assignment_kernel<DstTypeID, uint_kind, Src0TypeID, sint_kind, assign_error_overflow>
        : base_kernel<assignment_kernel<DstTypeID, uint_kind, Src0TypeID, sint_kind, assign_error_overflow>, 1> {
      typedef typename type_of<DstTypeID>::type dst_type;
      typedef typename type_of<Src0TypeID>::type src0_type;

      void single(char *dst, char *const *src)
      {
        src0_type s = *reinterpret_cast<src0_type *>(src[0]);

        DYND_TRACE_ASSIGNMENT(static_cast<dst_type>(s), dst_type, s, src_type);

        if (is_overflow<dst_type>(s)) {
          std::stringstream ss;
          ss << "overflow while assigning " << ndt::type::make<src0_type>() << " value ";
          ss << s << " to " << ndt::type::make<dst_type>();
          throw std::overflow_error(ss.str());
        }
        *reinterpret_cast<dst_type *>(dst) = static_cast<dst_type>(s);
      }
    };

    // Signed int -> unsigned int with other error checking
    template <type_id_t DstTypeID, type_id_t Src0TypeID>
    struct assignment_kernel<DstTypeID, uint_kind, Src0TypeID, sint_kind, assign_error_fractional>
        : assignment_kernel<DstTypeID, uint_kind, Src0TypeID, sint_kind, assign_error_overflow> {
    };

    template <type_id_t DstTypeID, type_id_t Src0TypeID>
    struct assignment_kernel<DstTypeID, uint_kind, Src0TypeID, sint_kind, assign_error_inexact>
        : assignment_kernel<DstTypeID, uint_kind, Src0TypeID, sint_kind, assign_error_overflow> {
    };

    // Unsigned int -> unsigned int with overflow checking just when sizeof(dst)
    // <
    // sizeof(src)
    template <type_id_t DstTypeID, type_id_t Src0TypeID>
    struct assignment_kernel<DstTypeID, uint_kind, Src0TypeID, uint_kind, assign_error_overflow>
        : base_kernel<assignment_kernel<DstTypeID, uint_kind, Src0TypeID, uint_kind, assign_error_overflow>, 1> {
      typedef typename type_of<DstTypeID>::type dst_type;
      typedef typename type_of<Src0TypeID>::type src0_type;

      void single(char *dst, char *const *src)
      {
        src0_type s = *reinterpret_cast<src0_type *>(src[0]);

        DYND_TRACE_ASSIGNMENT(static_cast<dst_type>(s), dst_type, s, src_type);

        if (is_overflow<dst_type>(s)) {
          std::stringstream ss;
          ss << "overflow while assigning " << ndt::type::make<src0_type>() << " value ";
          ss << s << " to " << ndt::type::make<dst_type>();
          throw std::overflow_error(ss.str());
        }
        *reinterpret_cast<dst_type *>(dst) = static_cast<dst_type>(s);
      }
    };

    // Unsigned int -> unsigned int with other error checking
    template <type_id_t DstTypeID, type_id_t Src0TypeID>
    struct assignment_kernel<DstTypeID, uint_kind, Src0TypeID, uint_kind, assign_error_fractional>
        : assignment_kernel<DstTypeID, uint_kind, Src0TypeID, uint_kind, assign_error_overflow> {
    };

    template <type_id_t DstTypeID, type_id_t Src0TypeID>
    struct assignment_kernel<DstTypeID, uint_kind, Src0TypeID, uint_kind, assign_error_inexact>
        : assignment_kernel<DstTypeID, uint_kind, Src0TypeID, uint_kind, assign_error_overflow> {
    };

    // Signed int -> floating point with inexact checking
    template <type_id_t DstTypeID, type_id_t Src0TypeID>
    struct assignment_kernel<DstTypeID, real_kind, Src0TypeID, sint_kind, assign_error_inexact>
        : base_kernel<assignment_kernel<DstTypeID, real_kind, Src0TypeID, sint_kind, assign_error_inexact>, 1> {
      typedef typename type_of<DstTypeID>::type dst_type;
      typedef typename type_of<Src0TypeID>::type src0_type;

      void single(char *dst, char *const *src)
      {
        src0_type s = *reinterpret_cast<src0_type *>(src[0]);
        dst_type d = static_cast<dst_type>(s);

        DYND_TRACE_ASSIGNMENT(d, dst_type, s, src0_type);

        if (static_cast<src0_type>(d) != s) {
          std::stringstream ss;
          ss << "inexact value while assigning " << ndt::type::make<src0_type>() << " value ";
          ss << s << " to " << ndt::type::make<dst_type>() << " value " << d;
          throw std::runtime_error(ss.str());
        }
        *reinterpret_cast<dst_type *>(dst) = d;
      }
    };

    // Signed int -> floating point with other checking
    template <type_id_t DstTypeID, type_id_t Src0TypeID>
    struct assignment_kernel<DstTypeID, real_kind, Src0TypeID, sint_kind, assign_error_overflow>
        : assignment_kernel<DstTypeID, real_kind, Src0TypeID, sint_kind, assign_error_nocheck> {
    };

    template <type_id_t DstTypeID, type_id_t Src0TypeID>
    struct assignment_kernel<DstTypeID, real_kind, Src0TypeID, sint_kind, assign_error_fractional>
        : assignment_kernel<DstTypeID, real_kind, Src0TypeID, sint_kind, assign_error_nocheck> {
    };

    template <type_id_t DstTypeID, type_id_t Src0TypeID, assign_error_mode ErrorMode>
    struct assignment_kernel<DstTypeID, complex_kind, Src0TypeID, real_kind, ErrorMode>
        : base_kernel<assignment_kernel<DstTypeID, complex_kind, Src0TypeID, real_kind, ErrorMode>, 1> {
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
    struct assignment_kernel<DstTypeID, real_kind, Src0TypeID, complex_kind, assign_error_overflow>
        : base_kernel<assignment_kernel<DstTypeID, real_kind, Src0TypeID, complex_kind, assign_error_overflow>, 1> {
      typedef typename type_of<DstTypeID>::type dst_type;
      typedef typename type_of<Src0TypeID>::type src0_type;

      void single(char *dst, char *const *src)
      {
        src0_type s = *reinterpret_cast<src0_type *>(src[0]);
        dst_type d;

        DYND_TRACE_ASSIGNMENT(static_cast<dst_type>(s.real()), dst_type, s, src0_type);

        if (s.imag() != 0) {
          std::stringstream ss;
          ss << "loss of imaginary component while assigning " << ndt::type::make<src0_type>() << " value ";
          ss << *src << " to " << ndt::type::make<dst_type>();
          throw std::runtime_error(ss.str());
        }

#if defined(DYND_USE_FPSTATUS)
        clear_fp_status();
        d = static_cast<dst_type>(s.real());
        if (is_overflow_fp_status()) {
          std::stringstream ss;
          ss << "overflow while assigning " << ndt::type::make<src0_type>() << " value ";
          ss << *src << " to " << ndt::type::make<dst_type>();
          throw std::overflow_error(ss.str());
        }
#else
        if (s.real() < -std::numeric_limits<dst_type>::max() || s.real() > std::numeric_limits<dst_type>::max()) {
          std::stringstream ss;
          ss << "overflow while assigning " << ndt::type::make<src0_type>() << " value ";
          ss << *src << " to " << ndt::type::make<dst_type>();
          throw std::overflow_error(ss.str());
        }
        d = static_cast<dst_type>(s.real());
#endif // DYND_USE_FPSTATUS

        *reinterpret_cast<dst_type *>(dst) = d;
      }
    };

    // complex -> real with inexact checking
    template <type_id_t DstTypeID, type_id_t Src0TypeID>
    struct assignment_kernel<DstTypeID, real_kind, Src0TypeID, complex_kind, assign_error_inexact>
        : assignment_kernel<DstTypeID, real_kind, Src0TypeID, complex_kind, assign_error_overflow> {
    };

    // complex -> real with fractional checking
    template <type_id_t DstTypeID, type_id_t SrcTypeID>
    struct assignment_kernel<DstTypeID, real_kind, SrcTypeID, complex_kind, assign_error_fractional>
        : assignment_kernel<DstTypeID, real_kind, SrcTypeID, complex_kind, assign_error_overflow> {
    };

    // complex<double> -> float with inexact checking
    template <>
    struct assignment_kernel<float32_type_id, real_kind, complex_float64_type_id, complex_kind, assign_error_inexact>
        : base_kernel<assignment_kernel<float32_type_id, real_kind, complex_float64_type_id, complex_kind,
                                        assign_error_inexact>,
                      1> {
      void single(char *dst, char *const *src)
      {
        complex<double> s = *reinterpret_cast<complex<double> *>(src[0]);
        float d;

        DYND_TRACE_ASSIGNMENT(static_cast<float>(s.real()), float, s, complex<double>);

        if (s.imag() != 0) {
          std::stringstream ss;
          ss << "loss of imaginary component while assigning " << ndt::type::make<complex<double>>() << " value ";
          ss << *src << " to " << ndt::type::make<float>();
          throw std::runtime_error(ss.str());
        }

#if defined(DYND_USE_FPSTATUS)
        clear_fp_status();
        d = static_cast<float>(s.real());
        if (is_overflow_fp_status()) {
          std::stringstream ss;
          ss << "overflow while assigning " << ndt::type::make<complex<double>>() << " value ";
          ss << s << " to " << ndt::type::make<float>();
          throw std::overflow_error(ss.str());
        }
#else
        if (s.real() < -std::numeric_limits<float>::max() || s.real() > std::numeric_limits<float>::max()) {
          std::stringstream ss;
          ss << "overflow while assigning " << ndt::type::make<complex<double>>() << " value ";
          ss << s << " to " << ndt::type::make<float>();
          throw std::overflow_error(ss.str());
        }
        d = static_cast<float>(s.real());
#endif // DYND_USE_FPSTATUS

        if (d != s.real()) {
          std::stringstream ss;
          ss << "inexact precision loss while assigning " << ndt::type::make<complex<double>>() << " value ";
          ss << *src << " to " << ndt::type::make<float>();
          throw std::runtime_error(ss.str());
        }

        *reinterpret_cast<float *>(dst) = d;
      }
    };

    // real -> complex with overflow checking
    template <type_id_t DstTypeID, type_id_t Src0TypeID>
    struct assignment_kernel<DstTypeID, complex_kind, Src0TypeID, real_kind, assign_error_overflow>
        : base_kernel<assignment_kernel<DstTypeID, complex_kind, Src0TypeID, real_kind, assign_error_overflow>, 1> {
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
          ss << "overflow while assigning " << ndt::type::make<src0_type>() << " value ";
          ss << s << " to " << ndt::type::make<dst_type>();
          throw std::overflow_error(ss.str());
        }
#else
        if (isfinite(s) && (s < -std::numeric_limits<typename dst_type::value_type>::max() ||
                            s > std::numeric_limits<typename dst_type::value_type>::max())) {
          std::stringstream ss;
          ss << "overflow while assigning " << ndt::type::make<src0_type>() << " value ";
          ss << s << " to " << ndt::type::make<dst_type>();
          throw std::overflow_error(ss.str());
        }
        d = static_cast<typename dst_type::value_type>(s);
#endif // DYND_USE_FPSTATUS

        *reinterpret_cast<dst_type *>(dst) = d;
      }
    };

    // real -> complex with fractional checking
    template <type_id_t DstTypeID, type_id_t Src0TypeID>
    struct assignment_kernel<DstTypeID, complex_kind, Src0TypeID, real_kind, assign_error_fractional>
        : assignment_kernel<DstTypeID, complex_kind, Src0TypeID, real_kind, assign_error_overflow> {
    };

    // real -> complex with inexact checking
    template <type_id_t DstTypeID, type_id_t Src0TypeID>
    struct assignment_kernel<DstTypeID, complex_kind, Src0TypeID, real_kind, assign_error_inexact>
        : base_kernel<assignment_kernel<DstTypeID, complex_kind, Src0TypeID, real_kind, assign_error_inexact>, 1> {
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
          ss << "overflow while assigning " << ndt::type::make<src0_type>() << " value ";
          ss << s << " to " << ndt::type::make<dst_type>();
          throw std::overflow_error(ss.str());
        }
#else
        if (isfinite(s) && (s < -std::numeric_limits<typename dst_type::value_type>::max() ||
                            s > std::numeric_limits<typename dst_type::value_type>::max())) {
          std::stringstream ss;
          ss << "overflow while assigning " << ndt::type::make<src0_type>() << " value ";
          ss << s << " to " << ndt::type::make<dst_type>();
          throw std::overflow_error(ss.str());
        }
        d = static_cast<typename dst_type::value_type>(s);
#endif // DYND_USE_FPSTATUS

        if (d != s) {
          std::stringstream ss;
          ss << "inexact precision loss while assigning " << ndt::type::make<src0_type>() << " value ";
          ss << s << " to " << ndt::type::make<dst_type>();
          throw std::runtime_error(ss.str());
        }

        *reinterpret_cast<dst_type *>(dst) = d;
      }
    };

    // complex<double> -> complex<float> with overflow checking
    template <>
    struct assignment_kernel<complex_float32_type_id, complex_kind, complex_float64_type_id, complex_kind,
                             assign_error_overflow>
        : base_kernel<assignment_kernel<complex_float32_type_id, complex_kind, complex_float64_type_id, complex_kind,
                                        assign_error_overflow>,
                      1> {
      typedef complex<float> dst_type;
      typedef complex<double> src0_type;

      void single(char *dst, char *const *src)
      {
        DYND_TRACE_ASSIGNMENT(static_cast<complex<float>>(*reinterpret_cast<src0_type *>(src[0])), complex<float>,
                              *reinterpret_cast<src0_type *>(src[0]), complex<double>);

#if defined(DYND_USE_FPSTATUS)
        clear_fp_status();
        *reinterpret_cast<dst_type *>(dst) = static_cast<complex<float>>(*reinterpret_cast<src0_type *>(src[0]));
        if (is_overflow_fp_status()) {
          std::stringstream ss;
          ss << "overflow while assigning " << ndt::type::make<complex<double>>() << " value ";
          ss << *src << " to " << ndt::type::make<complex<float>>();
          throw std::overflow_error(ss.str());
        }
#else
        complex<double>(s) = *reinterpret_cast<src0_type *>(src[0]);
        if (s.real() < -std::numeric_limits<float>::max() || s.real() > std::numeric_limits<float>::max() ||
            s.imag() < -std::numeric_limits<float>::max() || s.imag() > std::numeric_limits<float>::max()) {
          std::stringstream ss;
          ss << "overflow while assigning " << ndt::type::make<complex<double>>() << " value ";
          ss << s << " to " << ndt::type::make<complex<float>>();
          throw std::overflow_error(ss.str());
        }
        *reinterpret_cast<dst_type *>(dst) = static_cast<complex<float>>(s);
#endif // DYND_USE_FPSTATUS
      }
    };

    // complex<double> -> complex<float> with fractional checking
    template <>
    struct assignment_kernel<complex_float32_type_id, complex_kind, complex_float64_type_id, complex_kind,
                             assign_error_fractional>
        : assignment_kernel<complex_float32_type_id, complex_kind, complex_float64_type_id, complex_kind,
                            assign_error_overflow> {
    };

    // complex<double> -> complex<float> with inexact checking
    template <>
    struct assignment_kernel<complex_float32_type_id, complex_kind, complex_float64_type_id, complex_kind,
                             assign_error_inexact>
        : base_kernel<assignment_kernel<complex_float32_type_id, complex_kind, complex_float64_type_id, complex_kind,
                                        assign_error_inexact>,
                      1> {
      typedef complex<float> dst_type;
      typedef complex<double> src0_type;

      void single(char *dst, char *const *src)
      {
        DYND_TRACE_ASSIGNMENT(static_cast<complex<float>>(*reinterpret_cast<src0_type *>(src[0])), complex<float>,
                              *reinterpret_cast<src0_type *>(src[0]), complex<double>);

        complex<double> s = *reinterpret_cast<src0_type *>(src[0]);
        complex<float> d;

#if defined(DYND_USE_FPSTATUS)
        clear_fp_status();
        d = static_cast<complex<float>>(s);
        if (is_overflow_fp_status()) {
          std::stringstream ss;
          ss << "overflow while assigning " << ndt::type::make<complex<double>>() << " value ";
          ss << *reinterpret_cast<src0_type *>(src[0]) << " to " << ndt::type::make<complex<float>>();
          throw std::overflow_error(ss.str());
        }
#else
        if (s.real() < -std::numeric_limits<float>::max() || s.real() > std::numeric_limits<float>::max() ||
            s.imag() < -std::numeric_limits<float>::max() || s.imag() > std::numeric_limits<float>::max()) {
          std::stringstream ss;
          ss << "overflow while assigning " << ndt::type::make<complex<double>>() << " value ";
          ss << *reinterpret_cast<src0_type *>(src[0]) << " to " << ndt::type::make<complex<float>>();
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
          ss << "inexact precision loss while assigning " << ndt::type::make<complex<double>>() << " value ";
          ss << *reinterpret_cast<src0_type *>(src[0]) << " to " << ndt::type::make<complex<float>>();
          throw std::runtime_error(ss.str());
        }
        *reinterpret_cast<dst_type *>(dst) = d;
      }
    };

    template <assign_error_mode ErrorMode>
    struct assignment_kernel<date_type_id, datetime_kind, string_type_id, string_kind, ErrorMode>
        : base_kernel<assignment_kernel<date_type_id, datetime_kind, string_type_id, string_kind, ErrorMode>, 1> {
      ndt::type m_src_string_tp;
      const char *m_src_arrmeta;
      assign_error_mode m_errmode;
      date_parse_order_t m_date_parse_order;
      int m_century_window;

      assignment_kernel(const ndt::type &src_tp, const char *src_arrmeta, assign_error_mode errmode,
                        date_parse_order_t date_parse_order, int century_window)
          : m_src_string_tp(src_tp), m_src_arrmeta(src_arrmeta), m_errmode(errmode),
            m_date_parse_order(date_parse_order), m_century_window(century_window)
      {
      }

      void single(char *dst, char *const *src)
      {
        const ndt::base_string_type *bst = static_cast<const ndt::base_string_type *>(m_src_string_tp.extended());
        const std::string &s = bst->get_utf8_string(m_src_arrmeta, src[0], m_errmode);
        date_ymd ymd;
        // TODO: properly distinguish "date" and "option[date]" with respect to
        // NA support
        if (s == "NA") {
          ymd.set_to_na();
        }
        else {
          ymd.set_from_str(s, m_date_parse_order, m_century_window);
        }
        *reinterpret_cast<int32_t *>(dst) = ymd.to_days();
      }

      static intptr_t instantiate(char *DYND_UNUSED(static_data), char *DYND_UNUSED(data), void *ckb,
                                  intptr_t ckb_offset, const ndt::type &DYND_UNUSED(dst_tp),
                                  const char *DYND_UNUSED(dst_arrmeta), intptr_t DYND_UNUSED(nsrc),
                                  const ndt::type *src_tp, const char *const *src_arrmeta, kernel_request_t kernreq,
                                  const eval::eval_context *ectx, intptr_t DYND_UNUSED(nkwd),
                                  const nd::array *DYND_UNUSED(kwds),
                                  const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars))
      {
        assignment_kernel::make(ckb, kernreq, ckb_offset, src_tp[0], src_arrmeta[0], ectx->errmode,
                                ectx->date_parse_order, ectx->century_window);
        return ckb_offset;
      }
    };

    template <assign_error_mode ErrorMode>
    struct assignment_kernel<string_type_id, string_kind, date_type_id, datetime_kind, ErrorMode>
        : base_kernel<assignment_kernel<string_type_id, string_kind, date_type_id, datetime_kind, ErrorMode>, 1> {
      ndt::type m_dst_string_tp;
      const char *m_dst_arrmeta;
      eval::eval_context m_ectx;

      assignment_kernel(const ndt::type &dst_tp, const char *dst_arrmeta, const eval::eval_context *ectx)
          : m_dst_string_tp(dst_tp), m_dst_arrmeta(dst_arrmeta), m_ectx(*ectx)
      {
      }

      void single(char *dst, char *const *src)
      {
        date_ymd ymd;
        ymd.set_from_days(*reinterpret_cast<const int32_t *>(src[0]));
        std::string s = ymd.to_str();
        if (s.empty()) {
          s = "NA";
        }
        const ndt::base_string_type *bst = static_cast<const ndt::base_string_type *>(m_dst_string_tp.extended());
        bst->set_from_utf8_string(m_dst_arrmeta, dst, s, &m_ectx);
      }

      static intptr_t instantiate(char *DYND_UNUSED(static_data), char *DYND_UNUSED(data), void *ckb,
                                  intptr_t ckb_offset, const ndt::type &dst_tp, const char *dst_arrmeta,
                                  intptr_t DYND_UNUSED(nsrc), const ndt::type *DYND_UNUSED(src_tp),
                                  const char *const *DYND_UNUSED(src_arrmeta), kernel_request_t kernreq,
                                  const eval::eval_context *ectx, intptr_t DYND_UNUSED(nkwd),
                                  const nd::array *DYND_UNUSED(kwds),
                                  const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars))
      {
        assignment_kernel::make(ckb, kernreq, ckb_offset, dst_tp, dst_arrmeta, ectx);
        return ckb_offset;
      }
    };

    template <assign_error_mode ErrorMode>
    struct assignment_kernel<datetime_type_id, datetime_kind, string_type_id, string_kind, ErrorMode>
        : base_kernel<assignment_kernel<datetime_type_id, datetime_kind, string_type_id, string_kind, ErrorMode>, 1> {
      ndt::type m_dst_datetime_tp;
      ndt::type m_src_string_tp;
      const char *m_src_arrmeta;
      date_parse_order_t m_date_parse_order;
      int m_century_window;

      assignment_kernel(const ndt::type &dst_tp, const ndt::type &src_tp, const char *src_arrmeta,
                        date_parse_order_t date_parse_order, int century_window)
          : m_dst_datetime_tp(dst_tp), m_src_string_tp(src_tp), m_src_arrmeta(src_arrmeta),
            m_date_parse_order(date_parse_order), m_century_window(century_window)
      {
      }

      void single(char *dst, char *const *src)
      {
        const ndt::base_string_type *bst = static_cast<const ndt::base_string_type *>(m_src_string_tp.extended());
        const std::string &s = bst->get_utf8_string(m_src_arrmeta, src[0], ErrorMode);
        datetime_struct dts;
        // TODO: properly distinguish "date" and "option[date]" with respect to
        // NA
        // support
        if (s == "NA") {
          dts.set_to_na();
        }
        else {
          dts.set_from_str(s, m_date_parse_order, m_century_window);
        }
        *reinterpret_cast<int64_t *>(dst) = dts.to_ticks();
      }

      static intptr_t instantiate(char *DYND_UNUSED(static_data), char *DYND_UNUSED(data), void *ckb,
                                  intptr_t ckb_offset, const ndt::type &dst_tp, const char *DYND_UNUSED(dst_arrmeta),
                                  intptr_t DYND_UNUSED(nsrc), const ndt::type *src_tp, const char *const *src_arrmeta,
                                  kernel_request_t kernreq, const eval::eval_context *ectx, intptr_t DYND_UNUSED(nkwd),
                                  const nd::array *DYND_UNUSED(kwds),
                                  const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars))
      {
        assignment_kernel::make(ckb, kernreq, ckb_offset, dst_tp, src_tp[0], src_arrmeta[0], ectx->date_parse_order,
                                ectx->century_window);
        return ckb_offset;
      }
    };

    template <assign_error_mode ErrorMode>
    struct assignment_kernel<string_type_id, string_kind, datetime_type_id, datetime_kind, ErrorMode>
        : base_kernel<assignment_kernel<string_type_id, string_kind, datetime_type_id, datetime_kind, ErrorMode>, 1> {
      ndt::type m_dst_string_tp;
      const char *m_dst_arrmeta;
      ndt::type m_src_datetime_tp;
      eval::eval_context m_ectx;

      assignment_kernel(const ndt::type &dst_tp, const char *dst_arrmeta, const ndt::type &src_tp,
                        const eval::eval_context *ectx)
          : m_dst_string_tp(dst_tp), m_dst_arrmeta(dst_arrmeta), m_src_datetime_tp(src_tp), m_ectx(*ectx)
      {
      }

      void single(char *dst, char *const *src)
      {
        datetime_struct dts;
        dts.set_from_ticks(*reinterpret_cast<const int64_t *>(src[0]));
        std::string s = dts.to_str();
        if (s.empty()) {
          s = "NA";
        }
        else if (m_src_datetime_tp.extended<ndt::datetime_type>()->get_timezone() == tz_utc) {
          s += "Z";
        }
        const ndt::base_string_type *bst = static_cast<const ndt::base_string_type *>(m_dst_string_tp.extended());
        bst->set_from_utf8_string(m_dst_arrmeta, dst, s, &m_ectx);
      }

      static intptr_t instantiate(char *DYND_UNUSED(static_data), char *DYND_UNUSED(data), void *ckb,
                                  intptr_t ckb_offset, const ndt::type &dst_tp, const char *dst_arrmeta,
                                  intptr_t DYND_UNUSED(nsrc), const ndt::type *src_tp,
                                  const char *const *DYND_UNUSED(src_arrmeta), kernel_request_t kernreq,
                                  const eval::eval_context *ectx, intptr_t DYND_UNUSED(nkwd),
                                  const nd::array *DYND_UNUSED(kwds),
                                  const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars))
      {
        assignment_kernel::make(ckb, kernreq, ckb_offset, dst_tp, dst_arrmeta, src_tp[0], ectx);
        return ckb_offset;
      }
    };

    template <assign_error_mode ErrorMode>
    struct assignment_kernel<time_type_id, datetime_kind, string_type_id, string_kind, ErrorMode>
        : base_kernel<assignment_kernel<time_type_id, datetime_kind, string_type_id, string_kind, ErrorMode>, 1> {
      ndt::type m_src_string_tp;
      const char *m_src_arrmeta;
      assign_error_mode m_errmode;

      assignment_kernel(const ndt::type &src_tp, const char *src_arrmeta, assign_error_mode errmode)
          : m_src_string_tp(src_tp), m_src_arrmeta(src_arrmeta), m_errmode(errmode)
      {
      }

      void single(char *dst, char *const *src)
      {
        const ndt::base_string_type *bst = static_cast<const ndt::base_string_type *>(m_src_string_tp.extended());
        const std::string &s = bst->get_utf8_string(m_src_arrmeta, src[0], m_errmode);
        time_hmst hmst;
        // TODO: properly distinguish "time" and "option[time]" with respect to
        // NA
        // support
        if (s == "NA") {
          hmst.set_to_na();
        }
        else {
          hmst.set_from_str(s);
        }
        *reinterpret_cast<int64_t *>(dst) = hmst.to_ticks();
      }

      static intptr_t instantiate(char *DYND_UNUSED(static_data), char *DYND_UNUSED(data), void *ckb,
                                  intptr_t ckb_offset, const ndt::type &DYND_UNUSED(dst_tp),
                                  const char *DYND_UNUSED(dst_arrmeta), intptr_t DYND_UNUSED(nsrc),
                                  const ndt::type *src_tp, const char *const *src_arrmeta, kernel_request_t kernreq,
                                  const eval::eval_context *ectx, intptr_t DYND_UNUSED(nkwd),
                                  const nd::array *DYND_UNUSED(kwds),
                                  const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars))
      {
        assignment_kernel::make(ckb, kernreq, ckb_offset, src_tp[0], src_arrmeta[0], ectx->errmode);
        return ckb_offset;
      }
    };

    template <assign_error_mode ErrorMode>
    struct assignment_kernel<string_type_id, string_kind, time_type_id, datetime_kind, ErrorMode>
        : base_kernel<assignment_kernel<string_type_id, string_kind, time_type_id, datetime_kind, ErrorMode>, 1> {
      ndt::type m_dst_string_tp;
      const char *m_dst_arrmeta;
      eval::eval_context m_ectx;

      assignment_kernel(const ndt::type &dst_tp, const char *dst_arrmeta, const eval::eval_context *ectx)
          : m_dst_string_tp(dst_tp), m_dst_arrmeta(dst_arrmeta), m_ectx(*ectx)
      {
      }

      void single(char *dst, char *const *src)
      {
        time_hmst hmst;
        hmst.set_from_ticks(*reinterpret_cast<const int64_t *>(src[0]));
        std::string s = hmst.to_str();
        if (s.empty()) {
          s = "NA";
        }
        const ndt::base_string_type *bst = static_cast<const ndt::base_string_type *>(m_dst_string_tp.extended());
        bst->set_from_utf8_string(m_dst_arrmeta, dst, s, &m_ectx);
      }

      static intptr_t instantiate(char *DYND_UNUSED(static_data), char *DYND_UNUSED(data), void *ckb,
                                  intptr_t ckb_offset, const ndt::type &dst_tp, const char *dst_arrmeta,
                                  intptr_t DYND_UNUSED(nsrc), const ndt::type *DYND_UNUSED(src_tp),
                                  const char *const *DYND_UNUSED(src_arrmeta), kernel_request_t kernreq,
                                  const eval::eval_context *ectx, intptr_t DYND_UNUSED(nkwd),
                                  const nd::array *DYND_UNUSED(kwds),
                                  const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars))
      {
        assignment_kernel::make(ckb, kernreq, ckb_offset, dst_tp, dst_arrmeta, ectx);
        return ckb_offset;
      }
    };

    /**
     * A ckernel which assigns option[S] to option[T].
     */
    template <assign_error_mode ErrorMode>
    struct assignment_kernel<option_type_id, option_kind, option_type_id, option_kind, ErrorMode>
        : base_kernel<assignment_kernel<option_type_id, option_kind, option_type_id, option_kind, ErrorMode>, 1> {
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
        ckernel_prefix *src_is_avail = this->get_child();
        expr_single_t src_is_avail_fn = src_is_avail->get_function<expr_single_t>();
        bool1 avail = bool1(false);
        src_is_avail_fn(src_is_avail, reinterpret_cast<char *>(&avail), src);
        if (avail) {
          // It's available, copy using value assignment
          ckernel_prefix *value_assign = this->get_child(m_value_assign_offset);
          expr_single_t value_assign_fn = value_assign->get_function<expr_single_t>();
          value_assign_fn(value_assign, dst, src);
        }
        else {
          // It's not available, assign an NA
          ckernel_prefix *dst_assign_na = this->get_child(m_dst_assign_na_offset);
          expr_single_t dst_assign_na_fn = dst_assign_na->get_function<expr_single_t>();
          dst_assign_na_fn(dst_assign_na, dst, NULL);
        }
      }

      void strided(char *dst, intptr_t dst_stride, char *const *src, const intptr_t *src_stride, size_t count)
      {
        // Three child ckernels
        ckernel_prefix *src_is_avail = this->get_child();
        expr_strided_t src_is_avail_fn = src_is_avail->get_function<expr_strided_t>();
        ckernel_prefix *value_assign = this->get_child(m_value_assign_offset);
        expr_strided_t value_assign_fn = value_assign->get_function<expr_strided_t>();
        ckernel_prefix *dst_assign_na = this->get_child(m_dst_assign_na_offset);
        expr_strided_t dst_assign_na_fn = dst_assign_na->get_function<expr_strided_t>();
        // Process in chunks using the dynd default buffer size
        bool1 avail[DYND_BUFFER_CHUNK_SIZE];
        while (count > 0) {
          size_t chunk_size = std::min(count, (size_t)DYND_BUFFER_CHUNK_SIZE);
          count -= chunk_size;
          src_is_avail_fn(src_is_avail, reinterpret_cast<char *>(avail), 1, src, src_stride, chunk_size);
          void *avail_ptr = avail;
          char *src_copy = src[0];
          do {
            // Process a run of available values
            void *next_avail_ptr = memchr(avail_ptr, 0, chunk_size);
            if (!next_avail_ptr) {
              value_assign_fn(value_assign, dst, dst_stride, &src_copy, src_stride, chunk_size);
              dst += chunk_size * dst_stride;
              src += chunk_size * src_stride[0];
              break;
            }
            else if (next_avail_ptr > avail_ptr) {
              size_t segment_size = (char *)next_avail_ptr - (char *)avail_ptr;
              value_assign_fn(value_assign, dst, dst_stride, &src_copy, src_stride, segment_size);
              dst += segment_size * dst_stride;
              src_copy += segment_size * src_stride[0];
              chunk_size -= segment_size;
              avail_ptr = next_avail_ptr;
            }
            // Process a run of not available values
            next_avail_ptr = memchr(avail_ptr, 1, chunk_size);
            if (!next_avail_ptr) {
              dst_assign_na_fn(dst_assign_na, dst, dst_stride, NULL, NULL, chunk_size);
              dst += chunk_size * dst_stride;
              src_copy += chunk_size * src_stride[0];
              break;
            }
            else if (next_avail_ptr > avail_ptr) {
              size_t segment_size = (char *)next_avail_ptr - (char *)avail_ptr;
              dst_assign_na_fn(dst_assign_na, dst, dst_stride, NULL, NULL, segment_size);
              dst += segment_size * dst_stride;
              src_copy += segment_size * src_stride[0];
              chunk_size -= segment_size;
              avail_ptr = next_avail_ptr;
            }
          } while (chunk_size > 0);
        }
      }

      static intptr_t instantiate(char *DYND_UNUSED(static_data), char *DYND_UNUSED(data), void *ckb,
                                  intptr_t ckb_offset, const ndt::type &dst_tp, const char *dst_arrmeta, intptr_t nsrc,
                                  const ndt::type *src_tp, const char *const *src_arrmeta, kernel_request_t kernreq,
                                  const eval::eval_context *ectx, intptr_t nkwd, const nd::array *kwds,
                                  const std::map<std::string, ndt::type> &tp_vars)
      {
        intptr_t root_ckb_offset = ckb_offset;
        typedef assignment_kernel self_type;
        if (dst_tp.get_type_id() != option_type_id || src_tp[0].get_type_id() != option_type_id) {
          std::stringstream ss;
          ss << "option to option kernel needs option types, got " << dst_tp << " and " << src_tp[0];
          throw std::invalid_argument(ss.str());
        }
        const ndt::type &dst_val_tp = dst_tp.extended<ndt::option_type>()->get_value_type();
        const ndt::type &src_val_tp = src_tp[0].extended<ndt::option_type>()->get_value_type();
        self_type *self = self_type::make(ckb, kernreq, ckb_offset);
        // instantiate src_is_avail
        nd::callable &is_avail = src_tp[0].extended<ndt::option_type>()->get_is_avail();
        ckb_offset = is_avail.get()->instantiate(NULL, NULL, ckb, ckb_offset, ndt::type::make<bool1>(), NULL, nsrc,
                                                 src_tp, src_arrmeta, kernreq, ectx, nkwd, kwds, tp_vars);
        // instantiate dst_assign_na
        reinterpret_cast<ckernel_builder<kernel_request_host> *>(ckb)->reserve(ckb_offset + sizeof(ckernel_prefix));
        self = reinterpret_cast<ckernel_builder<kernel_request_host> *>(ckb)->get_at<self_type>(root_ckb_offset);
        self->m_dst_assign_na_offset = ckb_offset - root_ckb_offset;
        nd::callable &assign_na = dst_tp.extended<ndt::option_type>()->get_assign_na();
        ckb_offset = assign_na.get()->instantiate(NULL, NULL, ckb, ckb_offset, dst_tp, dst_arrmeta, nsrc, NULL, NULL,
                                                  kernreq, ectx, nkwd, kwds, tp_vars);
        // instantiate value_assign
        reinterpret_cast<ckernel_builder<kernel_request_host> *>(ckb)->reserve(ckb_offset + sizeof(ckernel_prefix));
        self = reinterpret_cast<ckernel_builder<kernel_request_host> *>(ckb)->get_at<self_type>(root_ckb_offset);
        self->m_value_assign_offset = ckb_offset - root_ckb_offset;
        ckb_offset =
            make_assignment_kernel(ckb, ckb_offset, dst_val_tp, dst_arrmeta, src_val_tp, src_arrmeta[0], kernreq, ectx);
        return ckb_offset;
      }
    };

    struct DYND_API string_to_option_bool_ck : nd::base_kernel<string_to_option_bool_ck, 1> {
      assign_error_mode m_errmode;

      void single(char *dst, char *const *src)
      {
        const string *std = reinterpret_cast<string *>(src[0]);
        parse::string_to_bool(dst, std->begin(), std->end(), true, m_errmode);
      }
    };

    struct DYND_API string_to_option_number_ck : nd::base_kernel<string_to_option_number_ck, 1> {
      type_id_t m_tid;
      assign_error_mode m_errmode;

      void single(char *dst, char *const *src)
      {
        const string *std = reinterpret_cast<string *>(src[0]);
        parse::string_to_number(dst, m_tid, std->begin(), std->end(), true, m_errmode);
      }
    };

    struct DYND_API string_to_option_tp_ck : nd::base_kernel<string_to_option_tp_ck, 1> {
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
        if (parse::matches_option_type_na_token(std->begin(), std->end())) {
          // It's not available, assign an NA
          ckernel_prefix *dst_assign_na = get_child(m_dst_assign_na_offset);
          expr_single_t dst_assign_na_fn = dst_assign_na->get_function<expr_single_t>();
          dst_assign_na_fn(dst_assign_na, dst, NULL);
        }
        else {
          // It's available, copy using value assignment
          ckernel_prefix *value_assign = get_child();
          expr_single_t value_assign_fn = value_assign->get_function<expr_single_t>();
          value_assign_fn(value_assign, dst, src);
        }
      }
    };

    template <type_id_t Src0TypeID, assign_error_mode ErrorMode>
    struct assignment_kernel<option_type_id, option_kind, Src0TypeID, real_kind, ErrorMode> {
      static intptr_t instantiate(char *DYND_UNUSED(static_data), char *DYND_UNUSED(data), void *ckb,
                                  intptr_t ckb_offset, const ndt::type &dst_tp, const char *dst_arrmeta, intptr_t nsrc,
                                  const ndt::type *src_tp, const char *const *src_arrmeta, kernel_request_t kernreq,
                                  const eval::eval_context *ectx, intptr_t nkwd, const nd::array *kwds,
                                  const std::map<std::string, ndt::type> &tp_vars)
      {
        // Deal with some float32 to option[T] conversions where any NaN is
        // interpreted
        // as NA.
        ndt::type src_tp_as_option = ndt::option_type::make(src_tp[0]);
        return assignment_kernel<option_type_id, option_kind, option_type_id, option_kind, ErrorMode>::instantiate(
            NULL, NULL, ckb, ckb_offset, dst_tp, dst_arrmeta, nsrc, &src_tp_as_option, src_arrmeta, kernreq, ectx, nkwd,
            kwds, tp_vars);
      }
    };

    template <assign_error_mode ErrorMode>
    struct assignment_kernel<option_type_id, option_kind, string_type_id, string_kind, ErrorMode> {
      static intptr_t instantiate(char *DYND_UNUSED(static_data), char *DYND_UNUSED(data), void *ckb,
                                  intptr_t ckb_offset, const ndt::type &dst_tp, const char *dst_arrmeta, intptr_t nsrc,
                                  const ndt::type *src_tp, const char *const *src_arrmeta, kernel_request_t kernreq,
                                  const eval::eval_context *ectx, intptr_t nkwd, const nd::array *kwds,
                                  const std::map<std::string, ndt::type> &tp_vars)
      {
        // Deal with some string to option[T] conversions where string values
        // might mean NA
        if (dst_tp.get_type_id() != option_type_id ||
            !(src_tp[0].get_kind() == string_kind ||
              (src_tp[0].get_type_id() == option_type_id &&
               src_tp[0].extended<ndt::option_type>()->get_value_type().get_kind() == string_kind))) {
          std::stringstream ss;
          ss << "string to option kernel needs string/option types, got (" << src_tp[0] << ") -> " << dst_tp;
          throw std::invalid_argument(ss.str());
        }

        type_id_t tid = dst_tp.extended<ndt::option_type>()->get_value_type().get_type_id();
        switch (tid) {
        case bool_type_id: {
          string_to_option_bool_ck *self = string_to_option_bool_ck::make(ckb, kernreq, ckb_offset);
          self->m_errmode = ectx->errmode;
          return ckb_offset;
        }
        case int8_type_id:
        case int16_type_id:
        case int32_type_id:
        case int64_type_id:
        case int128_type_id:
        case float16_type_id:
        case float32_type_id:
        case float64_type_id: {
          string_to_option_number_ck *self = string_to_option_number_ck::make(ckb, kernreq, ckb_offset);
          self->m_tid = tid;
          self->m_errmode = ectx->errmode;
          return ckb_offset;
        }
        case string_type_id: {
          // Just a string to string assignment
          return make_assignment_kernel(ckb, ckb_offset, dst_tp.extended<ndt::option_type>()->get_value_type(),
                                        dst_arrmeta, src_tp[0], src_arrmeta[0], kernreq, ectx);
        }
        default:
          break;
        }

        // Fall back to an adaptor that checks for a few standard
        // missing value tokens, then uses the standard value assignment
        intptr_t root_ckb_offset = ckb_offset;
        string_to_option_tp_ck *self = string_to_option_tp_ck::make(ckb, kernreq, ckb_offset);
        // First child ckernel is the value assignment
        ckb_offset = make_assignment_kernel(ckb, ckb_offset, dst_tp.extended<ndt::option_type>()->get_value_type(),
                                            dst_arrmeta, src_tp[0], src_arrmeta[0], kernreq, ectx);
        // Re-acquire self because the address may have changed
        self = reinterpret_cast<ckernel_builder<kernel_request_host> *>(ckb)
                   ->get_at<string_to_option_tp_ck>(root_ckb_offset);
        // Second child ckernel is the NA assignment
        self->m_dst_assign_na_offset = ckb_offset - root_ckb_offset;
        nd::callable &assign_na = dst_tp.extended<ndt::option_type>()->get_assign_na();
        ckb_offset = assign_na.get()->instantiate(NULL, NULL, ckb, ckb_offset, dst_tp, dst_arrmeta, nsrc, NULL, NULL,
                                                  kernreq, ectx, nkwd, kwds, tp_vars);
        return ckb_offset;
      }
    };

  } // namespace dynd::nd::detail

  /**
   * A ckernel which assigns option[S] to T.
   */
  struct DYND_API option_to_value_ck : nd::base_kernel<option_to_value_ck, 1> {
    // The default child is the src_is_avail ckernel
    size_t m_value_assign_offset;

    ~option_to_value_ck()
    {
      // src_is_avail
      get_child()->destroy();
      // value_assign
      get_child(m_value_assign_offset)->destroy();
    }

    void single(char *dst, char *const *src)
    {
      ckernel_prefix *src_is_avail = get_child();
      expr_single_t src_is_avail_fn = src_is_avail->get_function<expr_single_t>();
      ckernel_prefix *value_assign = get_child(m_value_assign_offset);
      expr_single_t value_assign_fn = value_assign->get_function<expr_single_t>();
      // Make sure it's not an NA
      bool1 avail = bool1(false);
      src_is_avail_fn(src_is_avail, reinterpret_cast<char *>(&avail), src);
      if (!avail) {
        throw std::overflow_error("cannot assign an NA value to a non-option type");
      }
      // Copy using value assignment
      value_assign_fn(value_assign, dst, src);
    }

    void strided(char *dst, intptr_t dst_stride, char *const *src, const intptr_t *src_stride, size_t count)
    {
      // Two child ckernels
      ckernel_prefix *src_is_avail = get_child();
      expr_strided_t src_is_avail_fn = src_is_avail->get_function<expr_strided_t>();
      ckernel_prefix *value_assign = get_child(m_value_assign_offset);
      expr_strided_t value_assign_fn = value_assign->get_function<expr_strided_t>();
      // Process in chunks using the dynd default buffer size
      bool1 avail[DYND_BUFFER_CHUNK_SIZE];
      char *src_copy = src[0];
      while (count > 0) {
        size_t chunk_size = std::min(count, (size_t)DYND_BUFFER_CHUNK_SIZE);
        src_is_avail_fn(src_is_avail, reinterpret_cast<char *>(avail), 1, &src_copy, src_stride, chunk_size);
        if (memchr(avail, 0, chunk_size) != NULL) {
          throw std::overflow_error("cannot assign an NA value to a non-option type");
        }
        value_assign_fn(value_assign, dst, dst_stride, &src_copy, src_stride, chunk_size);
        dst += chunk_size * dst_stride;
        src_copy += chunk_size * src_stride[0];
        count -= chunk_size;
      }
    }

    static intptr_t instantiate(char *DYND_UNUSED(static_data), char *DYND_UNUSED(data), void *ckb, intptr_t ckb_offset,
                                const ndt::type &dst_tp, const char *dst_arrmeta, intptr_t nsrc,
                                const ndt::type *src_tp, const char *const *src_arrmeta, kernel_request_t kernreq,
                                const eval::eval_context *ectx, intptr_t nkwd, const nd::array *kwds,
                                const std::map<std::string, ndt::type> &tp_vars)
    {
      intptr_t root_ckb_offset = ckb_offset;
      typedef dynd::nd::option_to_value_ck self_type;
      if (dst_tp.get_type_id() == option_type_id || src_tp[0].get_type_id() != option_type_id) {
        std::stringstream ss;
        ss << "option to value kernel needs value/option types, got " << dst_tp << " and " << src_tp[0];
        throw std::invalid_argument(ss.str());
      }
      const ndt::type &src_val_tp = src_tp[0].extended<ndt::option_type>()->get_value_type();
      self_type *self = self_type::make(ckb, kernreq, ckb_offset);
      // instantiate src_is_avail
      nd::callable &af = src_tp[0].extended<ndt::option_type>()->get_is_avail();
      ckb_offset = af.get()->instantiate(NULL, NULL, ckb, ckb_offset, ndt::type::make<bool1>(), NULL, nsrc, src_tp,
                                         src_arrmeta, kernreq, ectx, nkwd, kwds, tp_vars);
      // instantiate value_assign
      reinterpret_cast<ckernel_builder<kernel_request_host> *>(ckb)->reserve(ckb_offset + sizeof(ckernel_prefix));
      self = reinterpret_cast<ckernel_builder<kernel_request_host> *>(ckb)->get_at<self_type>(root_ckb_offset);
      self->m_value_assign_offset = ckb_offset - root_ckb_offset;
      return make_assignment_kernel(ckb, ckb_offset, dst_tp, dst_arrmeta, src_val_tp, src_arrmeta[0], kernreq, ectx);
    }
  };

  template <type_id_t DstTypeID, type_id_t Src0TypeID>
  using assignment_kernel = detail::assignment_virtual_kernel<DstTypeID, type_kind_of<DstTypeID>::value, Src0TypeID,
                                                              type_kind_of<Src0TypeID>::value>;

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

  struct DYND_API cuda_host_to_device_assign_ck : nd::expr_ck<cuda_host_to_device_assign_ck, kernel_request_host, 1> {
    size_t data_size;
    char *dst;

    cuda_host_to_device_assign_ck(size_t data_size) : data_size(data_size), dst(new char[data_size]) {}

    ~cuda_host_to_device_assign_ck() { delete[] dst; }

    void single(char *dst, char *const *src)
    {
      ckernel_prefix *child = this->get_child();
      expr_single_t single = child->get_function<expr_single_t>();

      single(this->dst, src, child);
      cuda_throw_if_not_success(cudaMemcpy(dst, this->dst, data_size, cudaMemcpyHostToDevice));
    }
  };

  struct DYND_API cuda_host_to_device_copy_ck : nd::expr_ck<cuda_host_to_device_copy_ck, kernel_request_host, 1> {
    size_t data_size;

    cuda_host_to_device_copy_ck(size_t data_size) : data_size(data_size) {}

    void single(char *dst, char *const *src)
    {
      cuda_throw_if_not_success(cudaMemcpy(dst, *src, data_size, cudaMemcpyHostToDevice));
    }
  };

  struct DYND_API cuda_device_to_host_assign_ck : nd::expr_ck<cuda_device_to_host_assign_ck, kernel_request_host, 1> {
    size_t data_size;
    char *src;

    cuda_device_to_host_assign_ck(size_t data_size) : data_size(data_size), src(new char[data_size]) {}

    ~cuda_device_to_host_assign_ck() { delete[] src; }

    void single(char *dst, char *const *src)
    {
      ckernel_prefix *child = this->get_child();
      expr_single_t single = child->get_function<expr_single_t>();

      cuda_throw_if_not_success(cudaMemcpy(this->src, *src, data_size, cudaMemcpyDeviceToHost));
      single(dst, &this->src, child);
    }
  };

  struct DYND_API cuda_device_to_host_copy_ck : nd::expr_ck<cuda_device_to_host_copy_ck, kernel_request_host, 1> {
    size_t data_size;

    cuda_device_to_host_copy_ck(size_t data_size) : data_size(data_size) {}

    void single(char *dst, char *const *src)
    {
      cuda_throw_if_not_success(cudaMemcpy(dst, *src, data_size, cudaMemcpyDeviceToHost));
    }
  };

  struct DYND_API cuda_device_to_device_copy_ck : nd::expr_ck<cuda_device_to_device_copy_ck, kernel_request_host, 1> {
    size_t data_size;

    cuda_device_to_device_copy_ck(size_t data_size) : data_size(data_size) {}

    void single(char *dst, char *const *src)
    {
      cuda_throw_if_not_success(cudaMemcpy(dst, *src, data_size, cudaMemcpyDeviceToDevice));
    }
  };

#endif

  template <class T>
  struct aligned_fixed_size_copy_assign_type : base_kernel<aligned_fixed_size_copy_assign_type<T>, 1> {
    void single(char *dst, char *const *src) { *reinterpret_cast<T *>(dst) = **reinterpret_cast<T *const *>(src); }

    void strided(char *dst, intptr_t dst_stride, char *const *src, const intptr_t *src_stride, size_t count)
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
  struct aligned_fixed_size_copy_assign<1> : base_kernel<aligned_fixed_size_copy_assign<1>, 1> {
    void single(char *dst, char *const *src) { *dst = **src; }

    void strided(char *dst, intptr_t dst_stride, char *const *src, const intptr_t *src_stride, size_t count)
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
  struct aligned_fixed_size_copy_assign<2> : aligned_fixed_size_copy_assign_type<int16_t> {
  };

  template <>
  struct aligned_fixed_size_copy_assign<4> : aligned_fixed_size_copy_assign_type<int32_t> {
  };

  template <>
  struct aligned_fixed_size_copy_assign<8> : aligned_fixed_size_copy_assign_type<int64_t> {
  };

  template <int N>
  struct unaligned_fixed_size_copy_assign : base_kernel<unaligned_fixed_size_copy_assign<N>, 1> {
    void single(char *dst, char *const *src) { memcpy(dst, *src, N); }

    void strided(char *dst, intptr_t dst_stride, char *const *src, const intptr_t *src_stride, size_t count)
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

  // All methods are inlined, so this can be public without declaring it DYND_API
  struct unaligned_copy_ck : base_kernel<unaligned_copy_ck, 1> {
    size_t data_size;

    unaligned_copy_ck(size_t data_size) : data_size(data_size) {}

    void single(char *dst, char *const *src) { memcpy(dst, *src, data_size); }

    void strided(char *dst, intptr_t dst_stride, char *const *src, const intptr_t *src_stride, size_t count)
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

  namespace detail {

    template <>
    struct assignment_virtual_kernel<date_type_id, datetime_kind, date_type_id, datetime_kind>
        : base_virtual_kernel<assignment_virtual_kernel<date_type_id, datetime_kind, date_type_id, datetime_kind>> {
      static intptr_t instantiate(char *DYND_UNUSED(static_data), char *DYND_UNUSED(data), void *ckb,
                                  intptr_t ckb_offset, const ndt::type &dst_tp, const char *DYND_UNUSED(dst_arrmeta),
                                  intptr_t DYND_UNUSED(nsrc), const ndt::type *DYND_UNUSED(src_tp),
                                  const char *const *DYND_UNUSED(src_arrmeta), kernel_request_t kernreq,
                                  const eval::eval_context *DYND_UNUSED(ectx), intptr_t DYND_UNUSED(nkwd),
                                  const nd::array *DYND_UNUSED(kwds),
                                  const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars))
      {
        return make_pod_typed_data_assignment_kernel(ckb, ckb_offset, dst_tp->get_data_size(),
                                                     dst_tp->get_data_alignment(), kernreq);
      }
    };

    template <>
    struct assignment_virtual_kernel<date_type_id, datetime_kind, struct_type_id, struct_kind>
        : base_virtual_kernel<assignment_virtual_kernel<date_type_id, datetime_kind, struct_type_id, struct_kind>> {
      static intptr_t instantiate(char *DYND_UNUSED(static_data), char *DYND_UNUSED(data), void *ckb,
                                  intptr_t ckb_offset, const ndt::type &dst_tp, const char *dst_arrmeta,
                                  intptr_t DYND_UNUSED(nsrc), const ndt::type *src_tp, const char *const *src_arrmeta,
                                  kernel_request_t kernreq, const eval::eval_context *ectx, intptr_t DYND_UNUSED(nkwd),
                                  const nd::array *DYND_UNUSED(kwds),
                                  const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars))
      {
        return make_assignment_kernel(ckb, ckb_offset, ndt::property_type::make(dst_tp, "struct"), dst_arrmeta, src_tp,
                                      src_arrmeta, kernreq, ectx);
      }
    };

    template <>
    struct assignment_virtual_kernel<struct_type_id, struct_kind, date_type_id, datetime_kind>
        : base_virtual_kernel<assignment_virtual_kernel<struct_type_id, struct_kind, date_type_id, datetime_kind>> {
      static intptr_t instantiate(char *DYND_UNUSED(static_data), char *DYND_UNUSED(data), void *ckb,
                                  intptr_t ckb_offset, const ndt::type &dst_tp, const char *dst_arrmeta,
                                  intptr_t DYND_UNUSED(nsrc), const ndt::type *src_tp, const char *const *src_arrmeta,
                                  kernel_request_t kernreq, const eval::eval_context *ectx, intptr_t DYND_UNUSED(nkwd),
                                  const nd::array *DYND_UNUSED(kwds),
                                  const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars))
      {
        return make_assignment_kernel(ckb, ckb_offset, dst_tp, dst_arrmeta,
                                      ndt::property_type::make(src_tp[0], "struct"), src_arrmeta[0], kernreq, ectx);
      }
    };

    template <>
    struct assignment_virtual_kernel<fixed_bytes_type_id, bytes_kind, fixed_bytes_type_id, bytes_kind>
        : base_virtual_kernel<
              assignment_virtual_kernel<fixed_bytes_type_id, bytes_kind, fixed_bytes_type_id, bytes_kind>> {
      static intptr_t instantiate(char *DYND_UNUSED(static_data), char *DYND_UNUSED(data), void *ckb,
                                  intptr_t ckb_offset, const ndt::type &dst_tp, const char *DYND_UNUSED(dst_arrmeta),
                                  intptr_t DYND_UNUSED(nsrc), const ndt::type *src_tp,
                                  const char *const *DYND_UNUSED(src_arrmeta), kernel_request_t kernreq,
                                  const eval::eval_context *DYND_UNUSED(ectx), intptr_t DYND_UNUSED(nkwd),
                                  const nd::array *DYND_UNUSED(kwds),
                                  const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars))
      {
        const ndt::fixed_bytes_type *src_fs = src_tp[0].extended<ndt::fixed_bytes_type>();
        if (dst_tp.get_data_size() != src_fs->get_data_size()) {
          throw std::runtime_error("cannot assign to a fixed_bytes type of a different size");
        }
        return make_pod_typed_data_assignment_kernel(
            ckb, ckb_offset, dst_tp.get_data_size(),
            std::min(dst_tp.get_data_alignment(), src_fs->get_data_alignment()), kernreq);
      }
    };

    template <type_id_t Src0TypeID>
    struct assignment_virtual_kernel<string_type_id, string_kind, Src0TypeID, sint_kind>
        : base_kernel<assignment_virtual_kernel<string_type_id, string_kind, Src0TypeID, sint_kind>, 1> {
      ndt::type dst_string_tp;
      type_id_t src_type_id;
      eval::eval_context ectx;
      const char *dst_arrmeta;

      assignment_virtual_kernel(const ndt::type &dst_string_tp, type_id_t src_type_id, eval::eval_context ectx,
                                const char *dst_arrmeta)
          : dst_string_tp(dst_string_tp), src_type_id(src_type_id), ectx(ectx), dst_arrmeta(dst_arrmeta)
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
        ndt::type(src_type_id).print_data(ss, NULL, src[0]);
        dst_string_tp->set_from_utf8_string(dst_arrmeta, dst, ss.str(), &ectx);
      }

      static intptr_t instantiate(char *DYND_UNUSED(static_data), char *DYND_UNUSED(data), void *ckb,
                                  intptr_t ckb_offset, const ndt::type &dst_tp, const char *dst_arrmeta,
                                  intptr_t DYND_UNUSED(nsrc), const ndt::type *src_tp,
                                  const char *const *DYND_UNUSED(src_arrmeta), kernel_request_t kernreq,
                                  const eval::eval_context *ectx, intptr_t DYND_UNUSED(nkwd),
                                  const nd::array *DYND_UNUSED(kwds),
                                  const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars))
      {
        assignment_virtual_kernel::make(ckb, kernreq, ckb_offset, dst_tp, src_tp[0].get_type_id(), *ectx, dst_arrmeta);
        return ckb_offset;
      }
    };

    template <type_id_t Src0TypeID, type_kind_t Src0TypeKind>
    struct assignment_virtual_kernel<fixed_string_type_id, string_kind, Src0TypeID, Src0TypeKind>
        : assignment_virtual_kernel<string_type_id, string_kind, int32_type_id, sint_kind> {
    };

    template <>
    struct assignment_virtual_kernel<tuple_type_id, tuple_kind, tuple_type_id, tuple_kind>
        : base_virtual_kernel<assignment_virtual_kernel<tuple_type_id, tuple_kind, tuple_type_id, tuple_kind>> {
      static intptr_t instantiate(char *DYND_UNUSED(static_data), char *DYND_UNUSED(data), void *ckb,
                                  intptr_t ckb_offset, const ndt::type &dst_tp, const char *dst_arrmeta,
                                  intptr_t DYND_UNUSED(nsrc), const ndt::type *src_tp, const char *const *src_arrmeta,
                                  kernel_request_t kernreq, const eval::eval_context *ectx, intptr_t DYND_UNUSED(nkwd),
                                  const nd::array *DYND_UNUSED(kwds),
                                  const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars))
      {
        if (dst_tp.extended() == src_tp[0].extended()) {
          return make_tuple_identical_assignment_kernel(ckb, ckb_offset, dst_tp, dst_arrmeta, src_arrmeta[0], kernreq,
                                                        ectx);
        }
        else if (src_tp[0].get_kind() == tuple_kind || src_tp[0].get_kind() == struct_kind) {
          return make_tuple_assignment_kernel(ckb, ckb_offset, dst_tp, dst_arrmeta, src_tp[0], src_arrmeta[0], kernreq,
                                              ectx);
        }
        else if (src_tp[0].is_builtin()) {
          return make_broadcast_to_tuple_assignment_kernel(ckb, ckb_offset, dst_tp, dst_arrmeta, src_tp[0],
                                                           src_arrmeta[0], kernreq, ectx);
        }

        std::stringstream ss;
        ss << "Cannot assign from " << src_tp[0] << " to " << dst_tp;
        throw dynd::type_error(ss.str());
      }
    };

    template <>
    struct assignment_virtual_kernel<struct_type_id, struct_kind, struct_type_id, struct_kind>
        : base_virtual_kernel<assignment_virtual_kernel<struct_type_id, struct_kind, struct_type_id, struct_kind>> {
      static intptr_t instantiate(char *DYND_UNUSED(static_data), char *DYND_UNUSED(data), void *ckb,
                                  intptr_t ckb_offset, const ndt::type &dst_tp, const char *dst_arrmeta,
                                  intptr_t DYND_UNUSED(nsrc), const ndt::type *src_tp, const char *const *src_arrmeta,
                                  kernel_request_t kernreq, const eval::eval_context *ectx, intptr_t DYND_UNUSED(nkwd),
                                  const nd::array *DYND_UNUSED(kwds),
                                  const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars))
      {
        if (dst_tp.extended() == src_tp[0].extended()) {
          return make_tuple_identical_assignment_kernel(ckb, ckb_offset, dst_tp, dst_arrmeta, src_arrmeta[0], kernreq,
                                                        ectx);
        }
        else if (src_tp[0].get_kind() == struct_kind) {
          return make_struct_assignment_kernel(ckb, ckb_offset, dst_tp, dst_arrmeta, src_tp[0], src_arrmeta[0], kernreq,
                                               ectx);
        }
        else if (src_tp[0].is_builtin()) {
          return make_broadcast_to_tuple_assignment_kernel(ckb, ckb_offset, dst_tp, dst_arrmeta, src_tp[0],
                                                           src_arrmeta[0], kernreq, ectx);
        }

        std::stringstream ss;
        ss << "Cannot assign from " << src_tp[0] << " to " << dst_tp;
        throw dynd::type_error(ss.str());
      }
    };

    template <>
    struct assignment_virtual_kernel<string_type_id, string_kind, fixed_string_type_id, string_kind>
        : base_kernel<assignment_virtual_kernel<string_type_id, string_kind, fixed_string_type_id, string_kind>, 1> {
      string_encoding_t m_dst_encoding, m_src_encoding;
      intptr_t m_src_element_size;
      next_unicode_codepoint_t m_next_fn;
      append_unicode_codepoint_t m_append_fn;

      assignment_virtual_kernel(string_encoding_t dst_encoding, string_encoding_t src_encoding,
                                intptr_t src_element_size, next_unicode_codepoint_t next_fn,
                                append_unicode_codepoint_t append_fn)
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

      static intptr_t instantiate(char *DYND_UNUSED(static_data), char *DYND_UNUSED(data), void *ckb,
                                  intptr_t ckb_offset, const ndt::type &dst_tp, const char *DYND_UNUSED(dst_arrmeta),
                                  intptr_t DYND_UNUSED(nsrc), const ndt::type *src_tp,
                                  const char *const *DYND_UNUSED(src_arrmeta), kernel_request_t kernreq,
                                  const eval::eval_context *ectx, intptr_t DYND_UNUSED(nkwd),
                                  const nd::array *DYND_UNUSED(kwds),
                                  const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars))
      {
        assignment_virtual_kernel::make(
            ckb, kernreq, ckb_offset, dst_tp.extended<ndt::base_string_type>()->get_encoding(),
            src_tp[0].extended<ndt::base_string_type>()->get_encoding(), src_tp[0].get_data_size(),
            get_next_unicode_codepoint_function(src_tp[0].extended<ndt::base_string_type>()->get_encoding(),
                                                ectx->errmode),
            get_append_unicode_codepoint_function(dst_tp.extended<ndt::base_string_type>()->get_encoding(),
                                                  ectx->errmode));
        return ckb_offset;
      }
    };

    template <type_id_t Src0TypeID, type_kind_t Src0TypeKind>
    struct assignment_virtual_kernel<adapt_type_id, expr_kind, Src0TypeID, Src0TypeKind>
        : base_virtual_kernel<assignment_virtual_kernel<adapt_type_id, expr_kind, Src0TypeID, Src0TypeKind>> {
      static intptr_t instantiate(char *DYND_UNUSED(static_data), char *DYND_UNUSED(data), void *ckb,
                                  intptr_t ckb_offset, const ndt::type &dst_tp, const char *dst_arrmeta,
                                  intptr_t DYND_UNUSED(nsrc), const ndt::type *src_tp, const char *const *src_arrmeta,
                                  kernel_request_t kernreq, const eval::eval_context *ectx, intptr_t DYND_UNUSED(nkwd),
                                  const nd::array *DYND_UNUSED(kwds),
                                  const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars))
      {
        return make_expression_assignment_kernel(ckb, ckb_offset, dst_tp, dst_arrmeta, src_tp[0], src_arrmeta[0],
                                                 kernreq, ectx);
      }
    };

    template <type_id_t Src0TypeID, type_kind_t Src0TypeKind>
    struct assignment_virtual_kernel<view_type_id, expr_kind, Src0TypeID, Src0TypeKind>
        : base_virtual_kernel<assignment_virtual_kernel<view_type_id, expr_kind, Src0TypeID, Src0TypeKind>> {
      static intptr_t instantiate(char *DYND_UNUSED(static_data), char *DYND_UNUSED(data), void *ckb,
                                  intptr_t ckb_offset, const ndt::type &dst_tp, const char *dst_arrmeta,
                                  intptr_t DYND_UNUSED(nsrc), const ndt::type *src_tp, const char *const *src_arrmeta,
                                  kernel_request_t kernreq, const eval::eval_context *ectx, intptr_t DYND_UNUSED(nkwd),
                                  const nd::array *DYND_UNUSED(kwds),
                                  const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars))
      {
        return make_expression_assignment_kernel(ckb, ckb_offset, dst_tp, dst_arrmeta, src_tp[0], src_arrmeta[0],
                                                 kernreq, ectx);
      }
    };

    template <>
    struct assignment_virtual_kernel<view_type_id, expr_kind, view_type_id, expr_kind>
        : base_virtual_kernel<assignment_virtual_kernel<view_type_id, expr_kind, view_type_id, expr_kind>> {
      static intptr_t instantiate(char *DYND_UNUSED(static_data), char *DYND_UNUSED(data), void *ckb,
                                  intptr_t ckb_offset, const ndt::type &dst_tp, const char *dst_arrmeta,
                                  intptr_t DYND_UNUSED(nsrc), const ndt::type *src_tp, const char *const *src_arrmeta,
                                  kernel_request_t kernreq, const eval::eval_context *ectx, intptr_t DYND_UNUSED(nkwd),
                                  const nd::array *DYND_UNUSED(kwds),
                                  const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars))
      {
        return make_expression_assignment_kernel(ckb, ckb_offset, dst_tp, dst_arrmeta, src_tp[0], src_arrmeta[0],
                                                 kernreq, ectx);
      }
    };

    template <>
    struct assignment_virtual_kernel<convert_type_id, expr_kind, convert_type_id, expr_kind>
        : base_virtual_kernel<assignment_virtual_kernel<convert_type_id, expr_kind, convert_type_id, expr_kind>> {
      static intptr_t instantiate(char *DYND_UNUSED(static_data), char *DYND_UNUSED(data), void *ckb,
                                  intptr_t ckb_offset, const ndt::type &dst_tp, const char *dst_arrmeta,
                                  intptr_t DYND_UNUSED(nsrc), const ndt::type *src_tp, const char *const *src_arrmeta,
                                  kernel_request_t kernreq, const eval::eval_context *ectx, intptr_t DYND_UNUSED(nkwd),
                                  const nd::array *DYND_UNUSED(kwds),
                                  const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars))
      {
        return make_expression_assignment_kernel(ckb, ckb_offset, dst_tp, dst_arrmeta, src_tp[0], src_arrmeta[0],
                                                 kernreq, ectx);
      }
    };

    template <type_id_t Src0TypeID, type_kind_t Src0TypeKind>
    struct assignment_virtual_kernel<convert_type_id, expr_kind, Src0TypeID, Src0TypeKind>
        : base_virtual_kernel<assignment_virtual_kernel<convert_type_id, expr_kind, Src0TypeID, Src0TypeKind>> {
      static intptr_t instantiate(char *DYND_UNUSED(static_data), char *DYND_UNUSED(data), void *ckb,
                                  intptr_t ckb_offset, const ndt::type &dst_tp, const char *dst_arrmeta,
                                  intptr_t DYND_UNUSED(nsrc), const ndt::type *src_tp, const char *const *src_arrmeta,
                                  kernel_request_t kernreq, const eval::eval_context *ectx, intptr_t DYND_UNUSED(nkwd),
                                  const nd::array *DYND_UNUSED(kwds),
                                  const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars))
      {
        return make_expression_assignment_kernel(ckb, ckb_offset, dst_tp, dst_arrmeta, src_tp[0], src_arrmeta[0],
                                                 kernreq, ectx);
      }
    };

    template <type_id_t Src0TypeID, type_kind_t Src0TypeKind>
    struct assignment_virtual_kernel<property_type_id, expr_kind, Src0TypeID, Src0TypeKind>
        : base_virtual_kernel<assignment_virtual_kernel<property_type_id, expr_kind, Src0TypeID, Src0TypeKind>> {
      static intptr_t instantiate(char *DYND_UNUSED(static_data), char *DYND_UNUSED(data), void *ckb,
                                  intptr_t ckb_offset, const ndt::type &dst_tp, const char *dst_arrmeta,
                                  intptr_t DYND_UNUSED(nsrc), const ndt::type *src_tp, const char *const *src_arrmeta,
                                  kernel_request_t kernreq, const eval::eval_context *ectx, intptr_t DYND_UNUSED(nkwd),
                                  const nd::array *DYND_UNUSED(kwds),
                                  const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars))
      {
        return make_expression_assignment_kernel(ckb, ckb_offset, dst_tp, dst_arrmeta, src_tp[0], src_arrmeta[0],
                                                 kernreq, ectx);
      }
    };

    template <type_id_t DstTypeID, type_kind_t DstKind, type_id_t Src0TypeID>
    struct assignment_virtual_kernel<DstTypeID, DstKind, Src0TypeID, expr_kind>
        : base_virtual_kernel<assignment_virtual_kernel<DstTypeID, DstKind, Src0TypeID, expr_kind>> {
      static intptr_t instantiate(char *DYND_UNUSED(static_data), char *DYND_UNUSED(data), void *ckb,
                                  intptr_t ckb_offset, const ndt::type &dst_tp, const char *dst_arrmeta,
                                  intptr_t DYND_UNUSED(nsrc), const ndt::type *src_tp, const char *const *src_arrmeta,
                                  kernel_request_t kernreq, const eval::eval_context *ectx, intptr_t DYND_UNUSED(nkwd),
                                  const nd::array *DYND_UNUSED(kwds),
                                  const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars))
      {
        return make_expression_assignment_kernel(ckb, ckb_offset, dst_tp, dst_arrmeta, src_tp[0], src_arrmeta[0],
                                                 kernreq, ectx);
      }
    };

    template <type_id_t Src0TypeID>
    struct assignment_virtual_kernel<fixed_string_type_id, string_kind, Src0TypeID, expr_kind>
        : base_virtual_kernel<assignment_virtual_kernel<fixed_string_type_id, string_kind, Src0TypeID, expr_kind>> {
      static intptr_t instantiate(char *DYND_UNUSED(static_data), char *DYND_UNUSED(data), void *ckb,
                                  intptr_t ckb_offset, const ndt::type &dst_tp, const char *dst_arrmeta,
                                  intptr_t DYND_UNUSED(nsrc), const ndt::type *src_tp, const char *const *src_arrmeta,
                                  kernel_request_t kernreq, const eval::eval_context *ectx, intptr_t DYND_UNUSED(nkwd),
                                  const nd::array *DYND_UNUSED(kwds),
                                  const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars))
      {
        return make_expression_assignment_kernel(ckb, ckb_offset, dst_tp, dst_arrmeta, src_tp[0], src_arrmeta[0],
                                                 kernreq, ectx);
      }
    };

    template <>
    struct assignment_virtual_kernel<bool_type_id, bool_kind, string_type_id, string_kind>
        : base_kernel<assignment_virtual_kernel<bool_type_id, bool_kind, string_type_id, string_kind>, 1> {
      ndt::type src_string_tp;
      assign_error_mode errmode;
      const char *src_arrmeta;

      assignment_virtual_kernel(const ndt::type &src_string_tp, assign_error_mode errmode, const char *src_arrmeta)
          : src_string_tp(src_string_tp), errmode(errmode), src_arrmeta(src_arrmeta)
      {
      }

      void single(char *dst, char *const *src)
      {
        // Get the string from the source
        std::string s = reinterpret_cast<const ndt::base_string_type *>(src_string_tp.extended())
                            ->get_utf8_string(src_arrmeta, src[0], errmode);
        trim(s);
        parse::string_to_bool(dst, s.data(), s.data() + s.size(), false, errmode);
      }

      static intptr_t instantiate(char *DYND_UNUSED(static_data), char *DYND_UNUSED(data), void *ckb,
                                  intptr_t ckb_offset, const ndt::type &DYND_UNUSED(dst_tp),
                                  const char *DYND_UNUSED(dst_arrmeta), intptr_t DYND_UNUSED(nsrc),
                                  const ndt::type *src_tp, const char *const *src_arrmeta, kernel_request_t kernreq,
                                  const eval::eval_context *ectx, intptr_t DYND_UNUSED(nkwd),
                                  const nd::array *DYND_UNUSED(kwds),
                                  const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars))
      {
        make(ckb, kernreq, ckb_offset, src_tp[0], ectx->errmode, src_arrmeta[0]);
        return ckb_offset;
      }
    };

    template <type_id_t Src0TypeID>
    struct assignment_virtual_kernel<Src0TypeID, sint_kind, string_type_id, string_kind>
        : base_kernel<assignment_virtual_kernel<Src0TypeID, sint_kind, string_type_id, string_kind>, 1> {
      typedef typename type_of<Src0TypeID>::type T;

      ndt::type src_string_tp;
      assign_error_mode errmode;
      const char *src_arrmeta;

      assignment_virtual_kernel(const ndt::type &src_string_tp, assign_error_mode errmode, const char *src_arrmeta)
          : src_string_tp(src_string_tp), errmode(errmode), src_arrmeta(src_arrmeta)
      {
      }

      void single(char *dst, char *const *src)
      {
        std::string s = reinterpret_cast<const ndt::base_string_type *>(src_string_tp.extended())
                            ->get_utf8_string(src_arrmeta, src[0], errmode);
        trim(s);
        bool negative = false;
        if (!s.empty() && s[0] == '-') {
          s.erase(0, 1);
          negative = true;
        }
        T result;
        if (errmode == assign_error_nocheck) {
          uint64_t value = parse::unchecked_string_to_uint64(s.data(), s.data() + s.size());
          result = negative ? static_cast<T>(-static_cast<int64_t>(value)) : static_cast<T>(value);
        }
        else {
          bool overflow = false, badparse = false;
          uint64_t value = parse::checked_string_to_uint64(s.data(), s.data() + s.size(), overflow, badparse);
          if (badparse) {
            raise_string_cast_error(ndt::type::make<T>(), src_string_tp, src_arrmeta, src[0]);
          }
          else if (overflow || overflow_check<T>::is_overflow(value, negative)) {
            raise_string_cast_overflow_error(ndt::type::make<T>(), src_string_tp, src_arrmeta, src[0]);
          }
          result = negative ? static_cast<T>(-static_cast<int64_t>(value)) : static_cast<T>(value);
        }
        *reinterpret_cast<T *>(dst) = result;
      }

      static intptr_t instantiate(char *DYND_UNUSED(static_data), char *DYND_UNUSED(data), void *ckb,
                                  intptr_t ckb_offset, const ndt::type &DYND_UNUSED(dst_tp),
                                  const char *DYND_UNUSED(dst_arrmeta), intptr_t DYND_UNUSED(nsrc),
                                  const ndt::type *src_tp, const char *const *src_arrmeta, kernel_request_t kernreq,
                                  const eval::eval_context *ectx, intptr_t DYND_UNUSED(nkwd),
                                  const nd::array *DYND_UNUSED(kwds),
                                  const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars))
      {
        assignment_virtual_kernel::make(ckb, kernreq, ckb_offset, src_tp[0], ectx->errmode, src_arrmeta[0]);
        return ckb_offset;
      }
    };

    template <>
    struct assignment_virtual_kernel<int128_type_id, sint_kind, string_type_id, string_kind>
        : base_kernel<assignment_virtual_kernel<int128_type_id, sint_kind, string_type_id, string_kind>, 1> {
      ndt::type src_string_tp;
      assign_error_mode errmode;
      const char *src_arrmeta;

      assignment_virtual_kernel(const ndt::type &src_string_tp, assign_error_mode errmode, const char *src_arrmeta)
          : src_string_tp(src_string_tp), errmode(errmode), src_arrmeta(src_arrmeta)
      {
      }

      void single(char *dst, char *const *src)
      {
        std::string s = reinterpret_cast<const ndt::base_string_type *>(src_string_tp.extended())
                            ->get_utf8_string(src_arrmeta, src[0], errmode);
        trim(s);
        bool negative = false;
        if (!s.empty() && s[0] == '-') {
          s.erase(0, 1);
          negative = true;
        }
        int128 result;
        if (errmode == assign_error_nocheck) {
          uint128 value = parse::unchecked_string_to_uint128(s.data(), s.data() + s.size());
          result = negative ? static_cast<int128>(0) : static_cast<int128>(value);
        }
        else {
          bool overflow = false, badparse = false;
          uint128 value = parse::checked_string_to_uint128(s.data(), s.data() + s.size(), overflow, badparse);
          if (badparse) {
            raise_string_cast_error(ndt::type::make<int128>(), src_string_tp, src_arrmeta, src[0]);
          }
          else if (overflow || overflow_check<int128>::is_overflow(value, negative)) {
            raise_string_cast_overflow_error(ndt::type::make<int128>(), src_string_tp, src_arrmeta, src[0]);
          }
          result = negative ? -static_cast<int128>(value) : static_cast<int128>(value);
        }
        *reinterpret_cast<int128 *>(dst) = result;
      }

      static intptr_t instantiate(char *DYND_UNUSED(static_data), char *DYND_UNUSED(data), void *ckb,
                                  intptr_t ckb_offset, const ndt::type &DYND_UNUSED(dst_tp),
                                  const char *DYND_UNUSED(dst_arrmeta), intptr_t DYND_UNUSED(nsrc),
                                  const ndt::type *src_tp, const char *const *src_arrmeta, kernel_request_t kernreq,
                                  const eval::eval_context *ectx, intptr_t DYND_UNUSED(nkwd),
                                  const nd::array *DYND_UNUSED(kwds),
                                  const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars))
      {
        make(ckb, kernreq, ckb_offset, src_tp[0], ectx->errmode, src_arrmeta[0]);
        return ckb_offset;
      }
    };

    template <type_id_t Src0TypeID>
    struct assignment_virtual_kernel<Src0TypeID, uint_kind, string_type_id, string_kind>
        : base_kernel<assignment_virtual_kernel<Src0TypeID, uint_kind, string_type_id, string_kind>, 1> {
      typedef typename type_of<Src0TypeID>::type T;

      ndt::type src_string_tp;
      assign_error_mode errmode;
      const char *src_arrmeta;

      assignment_virtual_kernel(const ndt::type &src_string_tp, assign_error_mode errmode, const char *src_arrmeta)
          : src_string_tp(src_string_tp), errmode(errmode), src_arrmeta(src_arrmeta)
      {
      }

      void single(char *dst, char *const *src)
      {
        std::string s = reinterpret_cast<const ndt::base_string_type *>(src_string_tp.extended())
                            ->get_utf8_string(src_arrmeta, src[0], errmode);
        trim(s);
        bool negative = false;
        if (!s.empty() && s[0] == '-') {
          s.erase(0, 1);
          negative = true;
        }
        T result;
        if (errmode == assign_error_nocheck) {
          uint64_t value = parse::unchecked_string_to_uint64(s.data(), s.data() + s.size());
          result = negative ? static_cast<T>(0) : static_cast<T>(value);
        }
        else {
          bool overflow = false, badparse = false;
          uint64_t value = parse::checked_string_to_uint64(s.data(), s.data() + s.size(), overflow, badparse);
          if (badparse) {
            raise_string_cast_error(ndt::type::make<T>(), src_string_tp, src_arrmeta, src[0]);
          }
          else if (overflow || (negative && value != 0) || overflow_check<T>::is_overflow(value)) {
            raise_string_cast_overflow_error(ndt::type::make<T>(), src_string_tp, src_arrmeta, src[0]);
          }
          result = static_cast<T>(value);
        }
        *reinterpret_cast<T *>(dst) = result;
      }

      static intptr_t instantiate(char *DYND_UNUSED(static_data), char *DYND_UNUSED(data), void *ckb,
                                  intptr_t ckb_offset, const ndt::type &DYND_UNUSED(dst_tp),
                                  const char *DYND_UNUSED(dst_arrmeta), intptr_t DYND_UNUSED(nsrc),
                                  const ndt::type *src_tp, const char *const *src_arrmeta, kernel_request_t kernreq,
                                  const eval::eval_context *ectx, intptr_t DYND_UNUSED(nkwd),
                                  const nd::array *DYND_UNUSED(kwds),
                                  const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars))
      {
        assignment_virtual_kernel::make(ckb, kernreq, ckb_offset, src_tp[0], ectx->errmode, src_arrmeta[0]);
        return ckb_offset;
      }
    };

    template <>
    struct assignment_virtual_kernel<uint128_type_id, uint_kind, string_type_id, string_kind>
        : base_kernel<assignment_virtual_kernel<uint128_type_id, uint_kind, string_type_id, string_kind>, 1> {
      ndt::type src_string_tp;
      assign_error_mode errmode;
      const char *src_arrmeta;

      assignment_virtual_kernel(const ndt::type &src_string_tp, assign_error_mode errmode, const char *src_arrmeta)
          : src_string_tp(src_string_tp), errmode(errmode), src_arrmeta(src_arrmeta)
      {
      }

      void single(char *dst, char *const *src)
      {
        std::string s = reinterpret_cast<const ndt::base_string_type *>(src_string_tp.extended())
                            ->get_utf8_string(src_arrmeta, src[0], errmode);
        trim(s);
        bool negative = false;
        if (!s.empty() && s[0] == '-') {
          s.erase(0, 1);
          negative = true;
        }
        int128 result;
        if (errmode == assign_error_nocheck) {
          result = parse::unchecked_string_to_uint128(s.data(), s.data() + s.size());
        }
        else {
          bool overflow = false, badparse = false;
          result = parse::checked_string_to_uint128(s.data(), s.data() + s.size(), overflow, badparse);
          if (badparse) {
            raise_string_cast_error(ndt::type::make<int128>(), src_string_tp, src_arrmeta, src[0]);
          }
          else if (overflow || (negative && result != 0)) {
            raise_string_cast_overflow_error(ndt::type::make<uint128>(), src_string_tp, src_arrmeta, src[0]);
          }
        }
        *reinterpret_cast<uint128 *>(dst) = result;
      }

      static intptr_t instantiate(char *DYND_UNUSED(static_data), char *DYND_UNUSED(data), void *ckb,
                                  intptr_t ckb_offset, const ndt::type &DYND_UNUSED(dst_tp),
                                  const char *DYND_UNUSED(dst_arrmeta), intptr_t DYND_UNUSED(nsrc),
                                  const ndt::type *src_tp, const char *const *src_arrmeta, kernel_request_t kernreq,
                                  const eval::eval_context *ectx, intptr_t DYND_UNUSED(nkwd),
                                  const nd::array *DYND_UNUSED(kwds),
                                  const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars))
      {
        make(ckb, kernreq, ckb_offset, src_tp[0], ectx->errmode, src_arrmeta[0]);
        return ckb_offset;
      }
    };

    template <>
    struct assignment_virtual_kernel<float16_type_id, real_kind, string_type_id, string_kind>
        : base_kernel<assignment_virtual_kernel<float16_type_id, real_kind, string_type_id, string_kind>, 1> {
      void single(char *DYND_UNUSED(dst), char *const *DYND_UNUSED(src))
      {
        throw std::runtime_error("TODO: implement string_to_float16_single");
      }
    };

    template <>
    struct assignment_virtual_kernel<float32_type_id, real_kind, string_type_id, string_kind>
        : base_kernel<assignment_virtual_kernel<float32_type_id, real_kind, string_type_id, string_kind>, 1> {
      ndt::type src_string_tp;
      assign_error_mode errmode;
      const char *src_arrmeta;

      assignment_virtual_kernel(const ndt::type &src_string_tp, assign_error_mode errmode, const char *src_arrmeta)
          : src_string_tp(src_string_tp), errmode(errmode), src_arrmeta(src_arrmeta)
      {
      }

      void single(char *dst, char *const *src)
      {
        // Get the string from the source
        std::string s = reinterpret_cast<const ndt::base_string_type *>(src_string_tp.extended())
                            ->get_utf8_string(src_arrmeta, src[0], errmode);
        trim(s);
        double value = parse::checked_string_to_float64(s.data(), s.data() + s.size(), errmode);
        // Assign double -> float according to the error mode
        char *child_src[1] = {reinterpret_cast<char *>(&value)};
        switch (errmode) {
        case assign_error_nocheck:
          dynd::nd::detail::assignment_kernel<float32_type_id, real_kind, float64_type_id, real_kind,
                                              assign_error_nocheck>::single_wrapper(NULL, dst, child_src);
          break;
        case assign_error_overflow:
          dynd::nd::detail::assignment_kernel<float32_type_id, real_kind, float64_type_id, real_kind,
                                              assign_error_overflow>::single_wrapper(NULL, dst, child_src);
          break;
        case assign_error_fractional:
          dynd::nd::detail::assignment_kernel<float32_type_id, real_kind, float64_type_id, real_kind,
                                              assign_error_fractional>::single_wrapper(NULL, dst, child_src);
          break;
        case assign_error_inexact:
          dynd::nd::detail::assignment_kernel<float32_type_id, real_kind, float64_type_id, real_kind,
                                              assign_error_inexact>::single_wrapper(NULL, dst, child_src);
          break;
        default:
          dynd::nd::detail::assignment_kernel<float32_type_id, real_kind, float64_type_id, real_kind,
                                              assign_error_fractional>::single_wrapper(NULL, dst, child_src);
          break;
        }
      }

      static intptr_t instantiate(char *DYND_UNUSED(static_data), char *DYND_UNUSED(data), void *ckb,
                                  intptr_t ckb_offset, const ndt::type &DYND_UNUSED(dst_tp),
                                  const char *DYND_UNUSED(dst_arrmeta), intptr_t DYND_UNUSED(nsrc),
                                  const ndt::type *src_tp, const char *const *src_arrmeta, kernel_request_t kernreq,
                                  const eval::eval_context *ectx, intptr_t DYND_UNUSED(nkwd),
                                  const nd::array *DYND_UNUSED(kwds),
                                  const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars))
      {
        make(ckb, kernreq, ckb_offset, src_tp[0], ectx->errmode, src_arrmeta[0]);
        return ckb_offset;
      }
    };

    template <>
    struct assignment_virtual_kernel<float64_type_id, real_kind, string_type_id, string_kind>
        : base_kernel<assignment_virtual_kernel<float64_type_id, real_kind, string_type_id, string_kind>, 1> {
      ndt::type src_string_tp;
      assign_error_mode errmode;
      const char *src_arrmeta;

      assignment_virtual_kernel(const ndt::type &src_string_tp, assign_error_mode errmode, const char *src_arrmeta)
          : src_string_tp(src_string_tp), errmode(errmode), src_arrmeta(src_arrmeta)
      {
      }

      void single(char *dst, char *const *src)
      {
        // Get the string from the source
        std::string s = reinterpret_cast<const ndt::base_string_type *>(src_string_tp.extended())
                            ->get_utf8_string(src_arrmeta, src[0], errmode);
        trim(s);
        double value = parse::checked_string_to_float64(s.data(), s.data() + s.size(), errmode);
        *reinterpret_cast<double *>(dst) = value;
      }

      static intptr_t instantiate(char *DYND_UNUSED(static_data), char *DYND_UNUSED(data), void *ckb,
                                  intptr_t ckb_offset, const ndt::type &DYND_UNUSED(dst_tp),
                                  const char *DYND_UNUSED(dst_arrmeta), intptr_t DYND_UNUSED(nsrc),
                                  const ndt::type *src_tp, const char *const *src_arrmeta, kernel_request_t kernreq,
                                  const eval::eval_context *ectx, intptr_t DYND_UNUSED(nkwd),
                                  const nd::array *DYND_UNUSED(kwds),
                                  const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars))
      {
        make(ckb, kernreq, ckb_offset, src_tp[0], ectx->errmode, src_arrmeta[0]);
        return ckb_offset;
      }
    };

    template <>
    struct assignment_virtual_kernel<float128_type_id, real_kind, string_type_id, string_kind>
        : base_kernel<assignment_virtual_kernel<float128_type_id, real_kind, string_type_id, string_kind>, 1> {
      void single(char *DYND_UNUSED(dst), char *const *DYND_UNUSED(src))
      {
        throw std::runtime_error("TODO: implement string_to_float128_single");
      }
    };

    template <>
    struct assignment_virtual_kernel<complex_float32_type_id, complex_kind, string_type_id, string_kind>
        : base_kernel<assignment_virtual_kernel<complex_float32_type_id, complex_kind, string_type_id, string_kind>,
                      1> {
      void single(char *DYND_UNUSED(dst), char *const *DYND_UNUSED(src))
      {
        throw std::runtime_error("TODO: implement string_to_complex_float32_single");
      }
    };

    template <>
    struct assignment_virtual_kernel<complex_float64_type_id, complex_kind, string_type_id, string_kind>
        : base_kernel<assignment_virtual_kernel<complex_float64_type_id, complex_kind, string_type_id, string_kind>,
                      1> {
      void single(char *DYND_UNUSED(dst), char *const *DYND_UNUSED(src))
      {
        throw std::runtime_error("TODO: implement string_to_complex_float64_single");
      }
    };

    template <type_id_t DstTypeID>
    struct assignment_virtual_kernel<DstTypeID, sint_kind, fixed_string_type_id, string_kind>
        : assignment_virtual_kernel<DstTypeID, sint_kind, string_type_id, string_kind> {
    };

    template <>
    struct assignment_virtual_kernel<fixed_string_type_id, string_kind, fixed_string_type_id, string_kind>
        : base_kernel<assignment_virtual_kernel<fixed_string_type_id, string_kind, fixed_string_type_id, string_kind>,
                      1> {
      next_unicode_codepoint_t m_next_fn;
      append_unicode_codepoint_t m_append_fn;
      intptr_t m_dst_data_size, m_src_data_size;
      bool m_overflow_check;

      assignment_virtual_kernel(next_unicode_codepoint_t next_fn, append_unicode_codepoint_t append_fn,
                                intptr_t dst_data_size, intptr_t src_data_size, bool overflow_check)
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

      static intptr_t instantiate(char *DYND_UNUSED(static_data), char *DYND_UNUSED(data), void *ckb,
                                  intptr_t ckb_offset, const ndt::type &dst_tp, const char *DYND_UNUSED(dst_arrmeta),
                                  intptr_t DYND_UNUSED(nsrc), const ndt::type *src_tp,
                                  const char *const *DYND_UNUSED(src_arrmeta), kernel_request_t kernreq,
                                  const eval::eval_context *ectx, intptr_t DYND_UNUSED(nkwd),
                                  const nd::array *DYND_UNUSED(kwds),
                                  const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars))
      {
        const ndt::fixed_string_type *src_fs = src_tp[0].extended<ndt::fixed_string_type>();
        assignment_virtual_kernel::make(
            ckb, kernreq, ckb_offset, get_next_unicode_codepoint_function(src_fs->get_encoding(), ectx->errmode),
            get_append_unicode_codepoint_function(dst_tp.extended<ndt::fixed_string_type>()->get_encoding(),
                                                  ectx->errmode),
            dst_tp.get_data_size(), src_fs->get_data_size(), ectx->errmode != assign_error_nocheck);
        return ckb_offset;
      }
    };

    template <>
    struct assignment_virtual_kernel<char_type_id, char_kind, char_type_id, char_kind>
        : assignment_virtual_kernel<fixed_string_type_id, string_kind, fixed_string_type_id, string_kind> {
    };

    template <>
    struct assignment_virtual_kernel<char_type_id, char_kind, fixed_string_type_id, string_kind>
        : assignment_virtual_kernel<fixed_string_type_id, string_kind, fixed_string_type_id, string_kind> {
    };

    struct new_adapt_assign_to_kernel : base_virtual_kernel<new_adapt_assign_to_kernel> {
      static intptr_t instantiate(char *DYND_UNUSED(static_data), char *data, void *ckb, intptr_t ckb_offset,
                                  const ndt::type &dst_tp, const char *dst_arrmeta, intptr_t nsrc,
                                  const ndt::type *DYND_UNUSED(src_tp), const char *const *src_arrmeta,
                                  kernel_request_t kernreq, const eval::eval_context *ectx, intptr_t nkwd,
                                  const nd::array *kwds, const std::map<std::string, ndt::type> &tp_vars)
      {
        const callable &inverse = dst_tp.extended<ndt::new_adapt_type>()->get_inverse();
        const ndt::type &value_tp = dst_tp.value_type();
        return inverse->instantiate(inverse->static_data(), data, ckb, ckb_offset, dst_tp.storage_type(), dst_arrmeta,
                                    nsrc, &value_tp, src_arrmeta, kernreq, ectx, nkwd, kwds, tp_vars);
      }
    };

    struct new_adapt_assign_from_kernel : base_kernel<new_adapt_assign_from_kernel, 1> {
      intptr_t forward_offset;

      void single(char *dst, char *const *src)
      {
        array buffer = empty(ndt::type("date"));

        get_child()->single(buffer.data(), src);

        char *child_src[1] = {buffer.data()};
        get_child(forward_offset)->single(dst, child_src);
      }

      static intptr_t instantiate(char *DYND_UNUSED(static_data), char *data, void *ckb, intptr_t ckb_offset,
                                  const ndt::type &dst_tp, const char *dst_arrmeta, intptr_t nsrc,
                                  const ndt::type *src_tp, const char *const *src_arrmeta, kernel_request_t kernreq,
                                  const eval::eval_context *ectx, intptr_t nkwd, const nd::array *kwds,
                                  const std::map<std::string, ndt::type> &tp_vars)
      {
        const callable &forward = src_tp[0].extended<ndt::new_adapt_type>()->get_forward();
        const ndt::type &storage_tp = src_tp[0].storage_type();

        intptr_t self_offset = ckb_offset;
        make(ckb, kernreq, ckb_offset);

        if (storage_tp.is_expression()) {
          ckb_offset = nd::assign::get()->instantiate(nd::assign::get()->static_data(), data, ckb, ckb_offset,
                                                      storage_tp.get_canonical_type(), dst_arrmeta, nsrc, &storage_tp,
                                                      src_arrmeta, kernel_request_single, ectx, nkwd, kwds, tp_vars);
        }

        intptr_t forward_offset = ckb_offset - self_offset;
        ndt::type src_tp2[1] = {storage_tp.get_canonical_type()};
        ckb_offset = forward->instantiate(forward->static_data(), data, ckb, ckb_offset, dst_tp, dst_arrmeta, nsrc,
                                          src_tp2, src_arrmeta, kernel_request_single, ectx, nkwd, kwds, tp_vars);
        get_self(reinterpret_cast<ckernel_builder<kernel_request_host> *>(ckb), self_offset)->forward_offset =
            forward_offset;

        return ckb_offset;
      }
    };

    template <>
    struct assignment_virtual_kernel<fixed_string_type_id, string_kind, string_type_id, string_kind>
        : base_kernel<assignment_virtual_kernel<fixed_string_type_id, string_kind, string_type_id, string_kind>, 1> {
      next_unicode_codepoint_t m_next_fn;
      append_unicode_codepoint_t m_append_fn;
      intptr_t m_dst_data_size;
      bool m_overflow_check;

      assignment_virtual_kernel(next_unicode_codepoint_t next_fn, append_unicode_codepoint_t append_fn,
                                intptr_t dst_data_size, bool overflow_check)
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

      static intptr_t instantiate(char *DYND_UNUSED(static_data), char *DYND_UNUSED(data), void *ckb,
                                  intptr_t ckb_offset, const ndt::type &dst_tp, const char *DYND_UNUSED(dst_arrmeta),
                                  intptr_t DYND_UNUSED(nsrc), const ndt::type *src_tp,
                                  const char *const *DYND_UNUSED(src_arrmeta), kernel_request_t kernreq,
                                  const eval::eval_context *ectx, intptr_t DYND_UNUSED(nkwd),
                                  const nd::array *DYND_UNUSED(kwds),
                                  const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars))
      {
        const ndt::base_string_type *src_fs = src_tp[0].extended<ndt::base_string_type>();
        assignment_virtual_kernel::make(ckb, kernreq, ckb_offset,
                                        get_next_unicode_codepoint_function(src_fs->get_encoding(), ectx->errmode),
                                        get_append_unicode_codepoint_function(
                                            dst_tp.extended<ndt::fixed_string_type>()->get_encoding(), ectx->errmode),
                                        dst_tp.get_data_size(), ectx->errmode != assign_error_nocheck);
        return ckb_offset;
      }
    };

    template <>
    struct assignment_virtual_kernel<char_type_id, char_kind, string_type_id, string_kind>
        : base_virtual_kernel<assignment_virtual_kernel<char_type_id, char_kind, string_type_id, string_kind>> {
      static intptr_t instantiate(char *DYND_UNUSED(static_data), char *DYND_UNUSED(data), void *ckb,
                                  intptr_t ckb_offset, const ndt::type &dst_tp, const char *DYND_UNUSED(dst_arrmeta),
                                  intptr_t DYND_UNUSED(nsrc), const ndt::type *src_tp,
                                  const char *const *DYND_UNUSED(src_arrmeta), kernel_request_t kernreq,
                                  const eval::eval_context *ectx, intptr_t DYND_UNUSED(nkwd),
                                  const nd::array *DYND_UNUSED(kwds),
                                  const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars))
      {
        const ndt::base_string_type *src_fs = src_tp[0].extended<ndt::base_string_type>();
        assignment_virtual_kernel<fixed_string_type_id, string_kind, string_type_id, string_kind>::make(
            ckb, kernreq, ckb_offset, get_next_unicode_codepoint_function(src_fs->get_encoding(), ectx->errmode),
            get_append_unicode_codepoint_function(dst_tp.extended<ndt::char_type>()->get_encoding(), ectx->errmode),
            dst_tp.get_data_size(), ectx->errmode != assign_error_nocheck);
        return ckb_offset;
      }
    };

    template <>
    struct assignment_virtual_kernel<fixed_string_type_id, string_kind, char_type_id, char_kind>
        : assignment_virtual_kernel<fixed_string_type_id, string_kind, fixed_string_type_id, string_kind> {
    };

    template <>
    struct assignment_virtual_kernel<string_type_id, string_kind, char_type_id, char_kind>
        : base_virtual_kernel<assignment_virtual_kernel<string_type_id, string_kind, char_type_id, char_kind>> {
      static intptr_t instantiate(char *DYND_UNUSED(static_data), char *DYND_UNUSED(data), void *ckb,
                                  intptr_t ckb_offset, const ndt::type &dst_tp, const char *DYND_UNUSED(dst_arrmeta),
                                  intptr_t DYND_UNUSED(nsrc), const ndt::type *src_tp,
                                  const char *const *DYND_UNUSED(src_arrmeta), kernel_request_t kernreq,
                                  const eval::eval_context *ectx, intptr_t DYND_UNUSED(nkwd),
                                  const nd::array *DYND_UNUSED(kwds),
                                  const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars))
      {
        assignment_virtual_kernel<string_type_id, string_kind, fixed_string_type_id, string_kind>::make(
            ckb, kernreq, ckb_offset, dst_tp.extended<ndt::base_string_type>()->get_encoding(),
            src_tp[0].extended<ndt::char_type>()->get_encoding(), src_tp[0].get_data_size(),
            get_next_unicode_codepoint_function(src_tp[0].extended<ndt::char_type>()->get_encoding(), ectx->errmode),
            get_append_unicode_codepoint_function(dst_tp.extended<ndt::base_string_type>()->get_encoding(),
                                                  ectx->errmode));
        return ckb_offset;
      }
    };

    template <>
    struct assignment_virtual_kernel<time_type_id, datetime_kind, struct_type_id, struct_kind>
        : base_virtual_kernel<assignment_virtual_kernel<time_type_id, datetime_kind, struct_type_id, struct_kind>> {
      static intptr_t instantiate(char *DYND_UNUSED(static_data), char *DYND_UNUSED(data), void *ckb,
                                  intptr_t ckb_offset, const ndt::type &dst_tp, const char *dst_arrmeta,
                                  intptr_t DYND_UNUSED(nsrc), const ndt::type *src_tp, const char *const *src_arrmeta,
                                  kernel_request_t kernreq, const eval::eval_context *ectx, intptr_t DYND_UNUSED(nkwd),
                                  const nd::array *DYND_UNUSED(kwds),
                                  const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars))
      {
        return make_assignment_kernel(ckb, ckb_offset, ndt::property_type::make(dst_tp, "struct"), dst_arrmeta,
                                      src_tp[0], src_arrmeta[0], kernreq, ectx);
      }
    };

    template <>
    struct assignment_virtual_kernel<struct_type_id, struct_kind, time_type_id, datetime_kind>
        : base_virtual_kernel<assignment_virtual_kernel<struct_type_id, struct_kind, time_type_id, datetime_kind>> {
      static intptr_t instantiate(char *DYND_UNUSED(static_data), char *DYND_UNUSED(data), void *ckb,
                                  intptr_t ckb_offset, const ndt::type &dst_tp, const char *dst_arrmeta,
                                  intptr_t DYND_UNUSED(nsrc), const ndt::type *src_tp, const char *const *src_arrmeta,
                                  kernel_request_t kernreq, const eval::eval_context *ectx, intptr_t DYND_UNUSED(nkwd),
                                  const nd::array *DYND_UNUSED(kwds),
                                  const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars))
      {
        // Convert to struct using the "struct" property
        return make_assignment_kernel(ckb, ckb_offset, dst_tp, dst_arrmeta,
                                      ndt::property_type::make(src_tp[0], "struct"), src_arrmeta[0], kernreq, ectx);
      }
    };

    template <>
    struct assignment_virtual_kernel<time_type_id, datetime_kind, time_type_id, datetime_kind>
        : base_virtual_kernel<assignment_virtual_kernel<time_type_id, datetime_kind, time_type_id, datetime_kind>> {
      static intptr_t instantiate(char *DYND_UNUSED(static_data), char *DYND_UNUSED(data), void *ckb,
                                  intptr_t ckb_offset, const ndt::type &dst_tp, const char *DYND_UNUSED(dst_arrmeta),
                                  intptr_t DYND_UNUSED(nsrc), const ndt::type *DYND_UNUSED(src_tp),
                                  const char *const *DYND_UNUSED(src_arrmeta), kernel_request_t kernreq,
                                  const eval::eval_context *DYND_UNUSED(ectx), intptr_t DYND_UNUSED(nkwd),
                                  const nd::array *DYND_UNUSED(kwds),
                                  const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars))
      {
        return make_pod_typed_data_assignment_kernel(ckb, ckb_offset, dst_tp.get_data_size(),
                                                     dst_tp.get_data_alignment(), kernreq);
      }
    };

    struct assignment_option_kernel : base_virtual_kernel<assignment_option_kernel> {
      static intptr_t instantiate(char *DYND_UNUSED(static_data), char *DYND_UNUSED(data), void *ckb,
                                  intptr_t ckb_offset, const ndt::type &dst_tp, const char *dst_arrmeta,
                                  intptr_t DYND_UNUSED(nsrc), const ndt::type *src_tp, const char *const *src_arrmeta,
                                  kernel_request_t kernreq, const eval::eval_context *ectx, intptr_t DYND_UNUSED(nkwd),
                                  const nd::array *DYND_UNUSED(kwds),
                                  const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars))
      {
        return kernels::make_option_assignment_kernel(ckb, ckb_offset, dst_tp, dst_arrmeta, src_tp[0], src_arrmeta[0],
                                                      kernreq, ectx);
      }
    };

    template <>
    struct assignment_virtual_kernel<struct_type_id, struct_kind, datetime_type_id, datetime_kind>
        : base_virtual_kernel<assignment_virtual_kernel<struct_type_id, struct_kind, datetime_type_id, datetime_kind>> {
      static intptr_t instantiate(char *DYND_UNUSED(static_data), char *DYND_UNUSED(data), void *ckb,
                                  intptr_t ckb_offset, const ndt::type &dst_tp, const char *dst_arrmeta,
                                  intptr_t DYND_UNUSED(nsrc), const ndt::type *src_tp, const char *const *src_arrmeta,
                                  kernel_request_t kernreq, const eval::eval_context *ectx, intptr_t DYND_UNUSED(nkwd),
                                  const nd::array *DYND_UNUSED(kwds),
                                  const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars))
      {
        return make_assignment_kernel(ckb, ckb_offset, dst_tp, dst_arrmeta,
                                      ndt::property_type::make(src_tp[0], "struct"), src_arrmeta[0], kernreq, ectx);
      }
    };

    template <>
    struct assignment_virtual_kernel<pointer_type_id, expr_kind, pointer_type_id, expr_kind>
        : base_virtual_kernel<assignment_virtual_kernel<pointer_type_id, expr_kind, pointer_type_id, expr_kind>> {
      static intptr_t instantiate(char *DYND_UNUSED(static_data), char *DYND_UNUSED(data), void *ckb,
                                  intptr_t ckb_offset, const ndt::type &dst_tp, const char *dst_arrmeta,
                                  intptr_t DYND_UNUSED(nsrc), const ndt::type *src_tp, const char *const *src_arrmeta,
                                  kernel_request_t kernreq, const eval::eval_context *ectx, intptr_t DYND_UNUSED(nkwd),
                                  const nd::array *DYND_UNUSED(kwds),
                                  const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars))
      {
        if (dst_tp == src_tp[0]) {
          return make_pod_typed_data_assignment_kernel(ckb, ckb_offset, dst_tp.get_data_size(),
                                                       dst_tp.get_data_alignment(), kernreq);
        }
        else {
          ndt::type dst_target_tp = dst_tp.extended<ndt::pointer_type>()->get_target_type();
          if (dst_target_tp == src_tp[0]) {
            return make_value_to_pointer_assignment_kernel(ckb, ckb_offset, dst_target_tp, kernreq);
          }
        }
        return make_expression_assignment_kernel(ckb, ckb_offset, dst_tp, dst_arrmeta, src_tp[0], src_arrmeta[0],
                                                 kernreq, ectx);
      }
    };

    template <>
    struct assignment_virtual_kernel<datetime_type_id, datetime_kind, struct_type_id, struct_kind>
        : base_virtual_kernel<assignment_virtual_kernel<datetime_type_id, datetime_kind, struct_type_id, struct_kind>> {
      static intptr_t instantiate(char *DYND_UNUSED(static_data), char *DYND_UNUSED(data), void *ckb,
                                  intptr_t ckb_offset, const ndt::type &dst_tp, const char *dst_arrmeta,
                                  intptr_t DYND_UNUSED(nsrc), const ndt::type *src_tp, const char *const *src_arrmeta,
                                  kernel_request_t kernreq, const eval::eval_context *ectx, intptr_t DYND_UNUSED(nkwd),
                                  const nd::array *DYND_UNUSED(kwds),
                                  const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars))
      {
        // Convert to struct using the "struct" property
        return make_assignment_kernel(ckb, ckb_offset, ndt::property_type::make(dst_tp, "struct"), dst_arrmeta,
                                      src_tp[0], src_arrmeta[0], kernreq, ectx);
      }
    };

    template <>
    struct assignment_virtual_kernel<datetime_type_id, datetime_kind, datetime_type_id, datetime_kind>
        : base_virtual_kernel<
              assignment_virtual_kernel<datetime_type_id, datetime_kind, datetime_type_id, datetime_kind>> {
      static intptr_t instantiate(char *DYND_UNUSED(static_data), char *DYND_UNUSED(data), void *ckb,
                                  intptr_t ckb_offset, const ndt::type &dst_tp, const char *DYND_UNUSED(dst_arrmeta),
                                  intptr_t DYND_UNUSED(nsrc), const ndt::type *src_tp,
                                  const char *const *DYND_UNUSED(src_arrmeta), kernel_request_t kernreq,
                                  const eval::eval_context *ectx, intptr_t DYND_UNUSED(nkwd),
                                  const nd::array *DYND_UNUSED(kwds),
                                  const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars))
      {
        if (src_tp[0].extended<ndt::datetime_type>()->get_timezone() == tz_abstract) {
          // TODO: If the destination timezone is not UTC, do an
          //       appropriate transformation
          if (dst_tp.extended<ndt::datetime_type>()->get_timezone() == tz_utc) {
            return make_pod_typed_data_assignment_kernel(ckb, ckb_offset, dst_tp.get_data_size(),
                                                         dst_tp.get_data_alignment(), kernreq);
          }
        }
        else if (dst_tp.extended<ndt::datetime_type>()->get_timezone() != tz_abstract) {
          // The value stored is independent of the time zone, so
          // a straight assignment is fine.
          return make_pod_typed_data_assignment_kernel(ckb, ckb_offset, dst_tp.get_data_size(),
                                                       dst_tp.get_data_alignment(), kernreq);
        }
        else if (ectx->errmode == assign_error_nocheck) {
          // TODO: If the source timezone is not UTC, do an appropriate
          //       transformation
          if (src_tp[0].extended<ndt::datetime_type>()->get_timezone() == tz_utc) {
            return make_pod_typed_data_assignment_kernel(ckb, ckb_offset, dst_tp.get_data_size(),
                                                         dst_tp.get_data_alignment(), kernreq);
          }
        }

        std::stringstream ss;
        ss << "Cannot assign from " << src_tp[0] << " to " << dst_tp;
        throw dynd::type_error(ss.str());
      }
    };

    template <>
    struct assignment_virtual_kernel<type_type_id, type_kind, type_type_id, type_kind>
        : base_kernel<assignment_virtual_kernel<type_type_id, type_kind, type_type_id, type_kind>, 1> {
      void single(char *dst, char *const *src)
      {
        *reinterpret_cast<ndt::type *>(dst) = *reinterpret_cast<ndt::type *>(src[0]);
      }
    };

  } // namespace dynd::nd::detail

  struct string_to_type_kernel : base_kernel<string_to_type_kernel, 1> {
    ndt::type src_string_dt;
    const char *src_arrmeta;
    assign_error_mode errmode;

    void single(char *dst, char *const *src)
    {
      const std::string &s =
          src_string_dt.extended<ndt::base_string_type>()->get_utf8_string(src_arrmeta, src[0], errmode);
      ndt::type(s).swap(*reinterpret_cast<ndt::type *>(dst));
    }

    static intptr_t instantiate(char *DYND_UNUSED(static_data), char *DYND_UNUSED(data), void *ckb, intptr_t ckb_offset,
                                const ndt::type &DYND_UNUSED(dst_tp), const char *DYND_UNUSED(dst_arrmeta),
                                intptr_t DYND_UNUSED(nsrc), const ndt::type *src_tp, const char *const *src_arrmeta,
                                kernel_request_t kernreq, const eval::eval_context *ectx, intptr_t DYND_UNUSED(nkwd),
                                const nd::array *DYND_UNUSED(kwds),
                                const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars))
    {
      string_to_type_kernel *e = make(ckb, kernreq, ckb_offset);
      // The kernel data owns a reference to this type
      e->src_string_dt = src_tp[0];
      e->src_arrmeta = src_arrmeta[0];
      e->errmode = ectx->errmode;
      return ckb_offset;
    }
  };

  struct type_to_string_kernel : base_kernel<type_to_string_kernel, 1> {
    ndt::type dst_string_dt;
    const char *dst_arrmeta;
    eval::eval_context ectx;

    void single(char *dst, char *const *src)
    {
      std::stringstream ss;
      ss << *reinterpret_cast<ndt::type *>(src[0]);
      dst_string_dt->set_from_utf8_string(dst_arrmeta, dst, ss.str(), &ectx);
    }

    static intptr_t instantiate(char *DYND_UNUSED(static_data), char *DYND_UNUSED(data), void *ckb, intptr_t ckb_offset,
                                const ndt::type &dst_tp, const char *dst_arrmeta, intptr_t DYND_UNUSED(nsrc),
                                const ndt::type *DYND_UNUSED(src_tp), const char *const *DYND_UNUSED(src_arrmeta),
                                kernel_request_t kernreq, const eval::eval_context *ectx, intptr_t DYND_UNUSED(nkwd),
                                const nd::array *DYND_UNUSED(kwds),
                                const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars))
    {
      // Type to string
      nd::type_to_string_kernel *e = nd::type_to_string_kernel::make(ckb, kernreq, ckb_offset);
      // The kernel data owns a reference to this type
      e->dst_string_dt = dst_tp;
      e->dst_arrmeta = dst_arrmeta;
      e->ectx = *ectx;
      return ckb_offset;
    }
  };

} // namespace dynd::nd

namespace ndt {

  template <type_id_t DstTypeID, type_id_t Src0TypeID>
  struct type::equivalent<nd::detail::assignment_virtual_kernel<DstTypeID, dynd::type_kind_of<DstTypeID>::value,
                                                                Src0TypeID, dynd::type_kind_of<Src0TypeID>::value>> {
    static type make() { return ndt::callable_type::make(type(DstTypeID), type(Src0TypeID)); }
  };

} // namespace dynd::ndt

} // namespace dynd
