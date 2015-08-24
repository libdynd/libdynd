//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/kernels/base_kernel.hpp>
#include <dynd/kernels/base_virtual_kernel.hpp>
#include <dynd/kernels/tuple_comparison_kernels.hpp>
#include <dynd/types/fixed_string_type.hpp>
#include <dynd/typed_data_assign.hpp>
#include <dynd/type.hpp>

namespace dynd {
namespace nd {

  template <typename K>
  struct base_comparison_kernel;

  template <template <type_id_t, type_id_t> class K, type_id_t I0, type_id_t I1>
  struct base_comparison_kernel<K<I0, I1>> : base_kernel<K<I0, I1>, 2> {
    static const std::size_t data_size = 0;
  };

  template <type_id_t I0, type_id_t I1>
  struct less_kernel : base_comparison_kernel<less_kernel<I0, I1>> {
    typedef typename type_of<I0>::type A0;
    typedef typename type_of<I1>::type A1;
    typedef typename std::common_type<A0, A1>::type T;

    void single(char *dst, char *const *src)
    {
      *reinterpret_cast<int *>(dst) =
          static_cast<T>(*reinterpret_cast<A0 *>(src[0])) <
          static_cast<T>(*reinterpret_cast<A1 *>(src[1]));
    }
  };

  template <type_id_t I0>
  struct less_kernel<I0, I0> : base_comparison_kernel<less_kernel<I0, I0>> {
    typedef typename type_of<I0>::type A0;

    void single(char *dst, char *const *src)
    {
      *reinterpret_cast<int *>(dst) =
          *reinterpret_cast<A0 *>(src[0]) < *reinterpret_cast<A0 *>(src[1]);
    }
  };

  template <string_encoding_t E>
  struct fixed_string_less_kernel;

  template <>
  struct
      fixed_string_less_kernel<string_encoding_utf_8> : base_kernel<
                                                            fixed_string_less_kernel<
                                                                string_encoding_utf_8>,
                                                            2> {
    size_t size;

    fixed_string_less_kernel(size_t size) : size(size)
    {
    }

    void single(char *dst, char *const *src)
    {
      *reinterpret_cast<int *>(dst) = strncmp(src[0], src[1], size) < 0;
    }

    static intptr_t instantiate(
        char *DYND_UNUSED(static_data), size_t DYND_UNUSED(data_size),
        char *DYND_UNUSED(data), void *ckb, intptr_t ckb_offset,
        const ndt::type &DYND_UNUSED(dst_tp),
        const char *DYND_UNUSED(dst_arrmeta), intptr_t DYND_UNUSED(nsrc),
        const ndt::type *src_tp, const char *const *DYND_UNUSED(src_arrmeta),
        kernel_request_t kernreq, const eval::eval_context *DYND_UNUSED(ectx),
        intptr_t DYND_UNUSED(nkwd), const nd::array *DYND_UNUSED(kwds),
        const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars))
    {
      make(ckb, kernreq, ckb_offset,
           src_tp[0].extended<ndt::fixed_string_type>()->get_size());
      return ckb_offset;
    }
  };

  template <>
  struct
      fixed_string_less_kernel<string_encoding_utf_16> : base_kernel<
                                                             fixed_string_less_kernel<
                                                                 string_encoding_utf_16>,
                                                             2> {
    size_t size;

    fixed_string_less_kernel(size_t size) : size(size)
    {
    }

    void single(char *dst, char *const *src)
    {
      *reinterpret_cast<int *>(dst) = std::lexicographical_compare(
          reinterpret_cast<const uint16_t *>(src[0]),
          reinterpret_cast<const uint16_t *>(src[0]) + size,
          reinterpret_cast<const uint16_t *>(src[1]),
          reinterpret_cast<const uint16_t *>(src[1]) + size);
    }

    static intptr_t instantiate(
        char *DYND_UNUSED(static_data), size_t DYND_UNUSED(data_size),
        char *DYND_UNUSED(data), void *ckb, intptr_t ckb_offset,
        const ndt::type &DYND_UNUSED(dst_tp),
        const char *DYND_UNUSED(dst_arrmeta), intptr_t DYND_UNUSED(nsrc),
        const ndt::type *src_tp, const char *const *DYND_UNUSED(src_arrmeta),
        kernel_request_t kernreq, const eval::eval_context *DYND_UNUSED(ectx),
        intptr_t DYND_UNUSED(nkwd), const nd::array *DYND_UNUSED(kwds),
        const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars))
    {
      make(ckb, kernreq, ckb_offset,
           src_tp[0].extended<ndt::fixed_string_type>()->get_size());
      return ckb_offset;
    }
  };

  template <>
  struct
      fixed_string_less_kernel<string_encoding_utf_32> : base_kernel<
                                                             fixed_string_less_kernel<
                                                                 string_encoding_utf_32>,
                                                             2> {
    size_t size;

    fixed_string_less_kernel(size_t size) : size(size)
    {
    }

    void single(char *dst, char *const *src)
    {
      *reinterpret_cast<int *>(dst) = std::lexicographical_compare(
          reinterpret_cast<const uint32_t *>(src[0]),
          reinterpret_cast<const uint32_t *>(src[0]) + size,
          reinterpret_cast<const uint32_t *>(src[1]),
          reinterpret_cast<const uint32_t *>(src[1]) + size);
    }

    static intptr_t instantiate(
        char *DYND_UNUSED(static_data), size_t DYND_UNUSED(data_size),
        char *DYND_UNUSED(data), void *ckb, intptr_t ckb_offset,
        const ndt::type &DYND_UNUSED(dst_tp),
        const char *DYND_UNUSED(dst_arrmeta), intptr_t DYND_UNUSED(nsrc),
        const ndt::type *src_tp, const char *const *DYND_UNUSED(src_arrmeta),
        kernel_request_t kernreq, const eval::eval_context *DYND_UNUSED(ectx),
        intptr_t DYND_UNUSED(nkwd), const nd::array *DYND_UNUSED(kwds),
        const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars))
    {
      make(ckb, kernreq, ckb_offset,
           src_tp[0].extended<ndt::fixed_string_type>()->get_size());
      return ckb_offset;
    }
  };

  template <>
  struct less_kernel<
      fixed_string_type_id,
      fixed_string_type_id> : base_virtual_kernel<less_kernel<fixed_string_type_id,
                                                              fixed_string_type_id>> {
    static intptr_t instantiate(
        char *static_data, size_t data_size, char *data, void *ckb,
        intptr_t ckb_offset, const ndt::type &dst_tp, const char *dst_arrmeta,
        intptr_t nsrc, const ndt::type *src_tp, const char *const *src_arrmeta,
        kernel_request_t kernreq, const eval::eval_context *ectx, intptr_t nkwd,
        const nd::array *kwds, const std::map<std::string, ndt::type> &tp_vars)
    {
      switch (src_tp[0].extended<ndt::fixed_string_type>()->get_encoding()) {
      case string_encoding_ascii:
      case string_encoding_utf_8:
        return fixed_string_less_kernel<string_encoding_utf_8>::instantiate(
            static_data, data_size, data, ckb, ckb_offset, dst_tp, dst_arrmeta,
            nsrc, src_tp, src_arrmeta, kernreq, ectx, nkwd, kwds, tp_vars);
      case string_encoding_utf_16:
        return fixed_string_less_kernel<string_encoding_utf_16>::instantiate(
            static_data, data_size, data, ckb, ckb_offset, dst_tp, dst_arrmeta,
            nsrc, src_tp, src_arrmeta, kernreq, ectx, nkwd, kwds, tp_vars);
      case string_encoding_utf_32:
        return fixed_string_less_kernel<string_encoding_utf_32>::instantiate(
            static_data, data_size, data, ckb, ckb_offset, dst_tp, dst_arrmeta,
            nsrc, src_tp, src_arrmeta, kernreq, ectx, nkwd, kwds, tp_vars);
      default:
        throw std::runtime_error("unidentified string encoding");
      }
    }
  };

  template <typename T>
  struct string_less_kernel : base_kernel<string_less_kernel<T>, 2> {
    void single(char *dst, char *const *src)
    {
      const string_type_data *da =
          reinterpret_cast<const string_type_data *>(src[0]);
      const string_type_data *db =
          reinterpret_cast<const string_type_data *>(src[1]);
      *reinterpret_cast<int *>(dst) =
          std::lexicographical_compare(reinterpret_cast<const T *>(da->begin),
                                       reinterpret_cast<const T *>(da->end),
                                       reinterpret_cast<const T *>(db->begin),
                                       reinterpret_cast<const T *>(db->end));
    }
  };

  template <>
  struct less_kernel<
      string_type_id,
      string_type_id> : base_virtual_kernel<less_kernel<string_type_id,
                                                        string_type_id>> {
    static intptr_t instantiate(
        char *static_data, size_t data_size, char *data, void *ckb,
        intptr_t ckb_offset, const ndt::type &dst_tp, const char *dst_arrmeta,
        intptr_t nsrc, const ndt::type *src_tp, const char *const *src_arrmeta,
        kernel_request_t kernreq, const eval::eval_context *ectx, intptr_t nkwd,
        const nd::array *kwds, const std::map<std::string, ndt::type> &tp_vars)
    {
      switch (src_tp[0].extended<ndt::string_type>()->get_encoding()) {
      case string_encoding_ascii:
      case string_encoding_utf_8:
        return string_less_kernel<uint8_t>::instantiate(
            static_data, data_size, data, ckb, ckb_offset, dst_tp, dst_arrmeta,
            nsrc, src_tp, src_arrmeta, kernreq, ectx, nkwd, kwds, tp_vars);
      case string_encoding_utf_16:
        return string_less_kernel<uint16_t>::instantiate(
            static_data, data_size, data, ckb, ckb_offset, dst_tp, dst_arrmeta,
            nsrc, src_tp, src_arrmeta, kernreq, ectx, nkwd, kwds, tp_vars);
      case string_encoding_utf_32:
        return string_less_kernel<uint32_t>::instantiate(
            static_data, data_size, data, ckb, ckb_offset, dst_tp, dst_arrmeta,
            nsrc, src_tp, src_arrmeta, kernreq, ectx, nkwd, kwds, tp_vars);
      default:
        throw std::runtime_error("unidentified string encoding");
      }
    }
  };

  template <type_id_t I0, type_id_t I1>
  struct less_equal_kernel : base_comparison_kernel<less_equal_kernel<I0, I1>> {
    typedef typename type_of<I0>::type A0;
    typedef typename type_of<I1>::type A1;
    typedef typename std::common_type<A0, A1>::type T;

    void single(char *dst, char *const *src)
    {
      *reinterpret_cast<int *>(dst) =
          static_cast<T>(*reinterpret_cast<A0 *>(src[0])) <=
          static_cast<T>(*reinterpret_cast<A1 *>(src[1]));
    }
  };

  template <type_id_t I0>
  struct less_equal_kernel<I0, I0> : base_comparison_kernel<
                                         less_equal_kernel<I0, I0>> {
    typedef typename type_of<I0>::type A0;

    void single(char *dst, char *const *src)
    {
      *reinterpret_cast<int *>(dst) =
          *reinterpret_cast<A0 *>(src[0]) <= *reinterpret_cast<A0 *>(src[1]);
    }
  };

  template <string_encoding_t E>
  struct fixed_string_less_equal_kernel;

  template <>
  struct fixed_string_less_equal_kernel<
      string_encoding_utf_8> : base_kernel<fixed_string_less_equal_kernel<string_encoding_utf_8>,
                                           2> {
    size_t size;

    fixed_string_less_equal_kernel(size_t size) : size(size)
    {
    }

    void single(char *dst, char *const *src)
    {
      *reinterpret_cast<int *>(dst) = strncmp(src[0], src[1], size) <= 0;
    }

    static intptr_t instantiate(
        char *DYND_UNUSED(static_data), size_t DYND_UNUSED(data_size),
        char *DYND_UNUSED(data), void *ckb, intptr_t ckb_offset,
        const ndt::type &DYND_UNUSED(dst_tp),
        const char *DYND_UNUSED(dst_arrmeta), intptr_t DYND_UNUSED(nsrc),
        const ndt::type *src_tp, const char *const *DYND_UNUSED(src_arrmeta),
        kernel_request_t kernreq, const eval::eval_context *DYND_UNUSED(ectx),
        intptr_t DYND_UNUSED(nkwd), const nd::array *DYND_UNUSED(kwds),
        const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars))
    {
      make(ckb, kernreq, ckb_offset,
           src_tp[0].extended<ndt::fixed_string_type>()->get_size());
      return ckb_offset;
    }
  };

  template <>
  struct fixed_string_less_equal_kernel<
      string_encoding_utf_16> : base_kernel<fixed_string_less_equal_kernel<string_encoding_utf_16>,
                                            2> {
    size_t size;

    fixed_string_less_equal_kernel(size_t size) : size(size)
    {
    }

    void single(char *dst, char *const *src)
    {
      *reinterpret_cast<int *>(dst) =
          !std::lexicographical_compare(
               reinterpret_cast<const uint16_t *>(src[1]),
               reinterpret_cast<const uint16_t *>(src[1]) + size,
               reinterpret_cast<const uint16_t *>(src[0]),
               reinterpret_cast<const uint16_t *>(src[0]) + size);
    }

    static intptr_t instantiate(
        char *DYND_UNUSED(static_data), size_t DYND_UNUSED(data_size),
        char *DYND_UNUSED(data), void *ckb, intptr_t ckb_offset,
        const ndt::type &DYND_UNUSED(dst_tp),
        const char *DYND_UNUSED(dst_arrmeta), intptr_t DYND_UNUSED(nsrc),
        const ndt::type *src_tp, const char *const *DYND_UNUSED(src_arrmeta),
        kernel_request_t kernreq, const eval::eval_context *DYND_UNUSED(ectx),
        intptr_t DYND_UNUSED(nkwd), const nd::array *DYND_UNUSED(kwds),
        const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars))
    {
      make(ckb, kernreq, ckb_offset,
           src_tp[0].extended<ndt::fixed_string_type>()->get_size());
      return ckb_offset;
    }
  };

  template <>
  struct fixed_string_less_equal_kernel<
      string_encoding_utf_32> : base_kernel<fixed_string_less_equal_kernel<string_encoding_utf_32>,
                                            2> {
    size_t size;

    fixed_string_less_equal_kernel(size_t size) : size(size)
    {
    }

    void single(char *dst, char *const *src)
    {
      *reinterpret_cast<int *>(dst) =
          !std::lexicographical_compare(
               reinterpret_cast<const uint32_t *>(src[1]),
               reinterpret_cast<const uint32_t *>(src[1]) + size,
               reinterpret_cast<const uint32_t *>(src[0]),
               reinterpret_cast<const uint32_t *>(src[0]) + size);
    }

    static intptr_t instantiate(
        char *DYND_UNUSED(static_data), size_t DYND_UNUSED(data_size),
        char *DYND_UNUSED(data), void *ckb, intptr_t ckb_offset,
        const ndt::type &DYND_UNUSED(dst_tp),
        const char *DYND_UNUSED(dst_arrmeta), intptr_t DYND_UNUSED(nsrc),
        const ndt::type *src_tp, const char *const *DYND_UNUSED(src_arrmeta),
        kernel_request_t kernreq, const eval::eval_context *DYND_UNUSED(ectx),
        intptr_t DYND_UNUSED(nkwd), const nd::array *DYND_UNUSED(kwds),
        const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars))
    {
      make(ckb, kernreq, ckb_offset,
           src_tp[0].extended<ndt::fixed_string_type>()->get_size());
      return ckb_offset;
    }
  };

  template <>
  struct less_equal_kernel<
      fixed_string_type_id,
      fixed_string_type_id> : base_virtual_kernel<less_equal_kernel<fixed_string_type_id,
                                                                    fixed_string_type_id>> {
    static intptr_t instantiate(
        char *static_data, size_t data_size, char *data, void *ckb,
        intptr_t ckb_offset, const ndt::type &dst_tp, const char *dst_arrmeta,
        intptr_t nsrc, const ndt::type *src_tp, const char *const *src_arrmeta,
        kernel_request_t kernreq, const eval::eval_context *ectx, intptr_t nkwd,
        const nd::array *kwds, const std::map<std::string, ndt::type> &tp_vars)
    {
      switch (src_tp[0].extended<ndt::fixed_string_type>()->get_encoding()) {
      case string_encoding_ascii:
      case string_encoding_utf_8:
        return fixed_string_less_equal_kernel<
            string_encoding_utf_8>::instantiate(static_data, data_size, data,
                                                ckb, ckb_offset, dst_tp,
                                                dst_arrmeta, nsrc, src_tp,
                                                src_arrmeta, kernreq, ectx,
                                                nkwd, kwds, tp_vars);
      case string_encoding_utf_16:
        return fixed_string_less_equal_kernel<
            string_encoding_utf_16>::instantiate(static_data, data_size, data,
                                                 ckb, ckb_offset, dst_tp,
                                                 dst_arrmeta, nsrc, src_tp,
                                                 src_arrmeta, kernreq, ectx,
                                                 nkwd, kwds, tp_vars);
      case string_encoding_utf_32:
        return fixed_string_less_equal_kernel<
            string_encoding_utf_32>::instantiate(static_data, data_size, data,
                                                 ckb, ckb_offset, dst_tp,
                                                 dst_arrmeta, nsrc, src_tp,
                                                 src_arrmeta, kernreq, ectx,
                                                 nkwd, kwds, tp_vars);
      default:
        throw std::runtime_error("unidentified string encoding");
      }
    }
  };

  template <typename T>
  struct string_less_equal_kernel
      : base_kernel<string_less_equal_kernel<T>, 2> {
    void single(char *dst, char *const *src)
    {
      const string_type_data *da =
          reinterpret_cast<const string_type_data *>(src[0]);
      const string_type_data *db =
          reinterpret_cast<const string_type_data *>(src[1]);
      *reinterpret_cast<int *>(dst) =
          !std::lexicographical_compare(reinterpret_cast<const T *>(db->begin),
                                        reinterpret_cast<const T *>(db->end),
                                        reinterpret_cast<const T *>(da->begin),
                                        reinterpret_cast<const T *>(da->end));
    }
  };

  template <>
  struct less_equal_kernel<
      string_type_id,
      string_type_id> : base_virtual_kernel<less_equal_kernel<string_type_id,
                                                              string_type_id>> {
    static intptr_t instantiate(
        char *static_data, size_t data_size, char *data, void *ckb,
        intptr_t ckb_offset, const ndt::type &dst_tp, const char *dst_arrmeta,
        intptr_t nsrc, const ndt::type *src_tp, const char *const *src_arrmeta,
        kernel_request_t kernreq, const eval::eval_context *ectx, intptr_t nkwd,
        const nd::array *kwds, const std::map<std::string, ndt::type> &tp_vars)
    {
      switch (src_tp[0].extended<ndt::string_type>()->get_encoding()) {
      case string_encoding_ascii:
      case string_encoding_utf_8:
        return string_less_equal_kernel<uint8_t>::instantiate(
            static_data, data_size, data, ckb, ckb_offset, dst_tp, dst_arrmeta,
            nsrc, src_tp, src_arrmeta, kernreq, ectx, nkwd, kwds, tp_vars);
      case string_encoding_utf_16:
        return string_less_equal_kernel<uint16_t>::instantiate(
            static_data, data_size, data, ckb, ckb_offset, dst_tp, dst_arrmeta,
            nsrc, src_tp, src_arrmeta, kernreq, ectx, nkwd, kwds, tp_vars);
      case string_encoding_utf_32:
        return string_less_equal_kernel<uint32_t>::instantiate(
            static_data, data_size, data, ckb, ckb_offset, dst_tp, dst_arrmeta,
            nsrc, src_tp, src_arrmeta, kernreq, ectx, nkwd, kwds, tp_vars);
      default:
        throw std::runtime_error("unidentified string encoding");
      }
    }
  };

  template <type_id_t I0, type_id_t I1>
  struct equal_kernel : base_comparison_kernel<equal_kernel<I0, I1>> {
    typedef typename type_of<I0>::type A0;
    typedef typename type_of<I1>::type A1;
    typedef typename std::common_type<A0, A1>::type T;

    void single(char *dst, char *const *src)
    {
      *reinterpret_cast<int *>(dst) =
          static_cast<T>(*reinterpret_cast<A0 *>(src[0])) ==
          static_cast<T>(*reinterpret_cast<A1 *>(src[1]));
    }
  };

  template <type_id_t I0>
  struct equal_kernel<I0, I0> : base_comparison_kernel<equal_kernel<I0, I0>> {
    typedef typename type_of<I0>::type A0;

    void single(char *dst, char *const *src)
    {
      *reinterpret_cast<int *>(dst) =
          *reinterpret_cast<A0 *>(src[0]) == *reinterpret_cast<A0 *>(src[1]);
    }
  };

  template <typename T>
  struct string_equal_kernel : base_kernel<string_equal_kernel<T>, 2> {
    void single(char *dst, char *const *src)
    {
      const string_type_data *da =
          reinterpret_cast<const string_type_data *>(src[0]);
      const string_type_data *db =
          reinterpret_cast<const string_type_data *>(src[1]);
      *reinterpret_cast<int *>(dst) =
          (da->end - da->begin == db->end - db->begin) &&
          memcmp(da->begin, db->begin, da->end - da->begin) == 0;
    }
  };

  template <>
  struct equal_kernel<
      string_type_id,
      string_type_id> : base_virtual_kernel<equal_kernel<string_type_id,
                                                         string_type_id>> {
    static intptr_t instantiate(
        char *static_data, size_t data_size, char *data, void *ckb,
        intptr_t ckb_offset, const ndt::type &dst_tp, const char *dst_arrmeta,
        intptr_t nsrc, const ndt::type *src_tp, const char *const *src_arrmeta,
        kernel_request_t kernreq, const eval::eval_context *ectx, intptr_t nkwd,
        const nd::array *kwds, const std::map<std::string, ndt::type> &tp_vars)
    {
      switch (src_tp[0].extended<ndt::string_type>()->get_encoding()) {
      case string_encoding_ascii:
      case string_encoding_utf_8:
        return string_equal_kernel<uint8_t>::instantiate(
            static_data, data_size, data, ckb, ckb_offset, dst_tp, dst_arrmeta,
            nsrc, src_tp, src_arrmeta, kernreq, ectx, nkwd, kwds, tp_vars);
      case string_encoding_utf_16:
        return string_equal_kernel<uint16_t>::instantiate(
            static_data, data_size, data, ckb, ckb_offset, dst_tp, dst_arrmeta,
            nsrc, src_tp, src_arrmeta, kernreq, ectx, nkwd, kwds, tp_vars);
      case string_encoding_utf_32:
        return string_equal_kernel<uint32_t>::instantiate(
            static_data, data_size, data, ckb, ckb_offset, dst_tp, dst_arrmeta,
            nsrc, src_tp, src_arrmeta, kernreq, ectx, nkwd, kwds, tp_vars);
      default:
        throw std::runtime_error("unidentified string encoding");
      }
    }
  };

  template <string_encoding_t E>
  struct fixed_string_equal_kernel;

  template <>
  struct fixed_string_equal_kernel<
      string_encoding_utf_8> : base_kernel<fixed_string_equal_kernel<string_encoding_utf_8>,
                                           2> {
    size_t size;

    fixed_string_equal_kernel(size_t size) : size(size)
    {
    }

    void single(char *dst, char *const *src)
    {
      *reinterpret_cast<int *>(dst) = strncmp(src[0], src[1], size) == 0;
    }

    static intptr_t instantiate(
        char *DYND_UNUSED(static_data), size_t DYND_UNUSED(data_size),
        char *DYND_UNUSED(data), void *ckb, intptr_t ckb_offset,
        const ndt::type &DYND_UNUSED(dst_tp),
        const char *DYND_UNUSED(dst_arrmeta), intptr_t DYND_UNUSED(nsrc),
        const ndt::type *src_tp, const char *const *DYND_UNUSED(src_arrmeta),
        kernel_request_t kernreq, const eval::eval_context *DYND_UNUSED(ectx),
        intptr_t DYND_UNUSED(nkwd), const nd::array *DYND_UNUSED(kwds),
        const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars))
    {
      make(ckb, kernreq, ckb_offset,
           src_tp[0].extended<ndt::fixed_string_type>()->get_size());
      return ckb_offset;
    }
  };

  template <>
  struct fixed_string_equal_kernel<
      string_encoding_utf_16> : base_kernel<fixed_string_equal_kernel<string_encoding_utf_16>,
                                            2> {
    size_t size;

    fixed_string_equal_kernel(size_t size) : size(size)
    {
    }

    void single(char *dst, char *const *src)
    {
      const uint16_t *lhs = reinterpret_cast<const uint16_t *>(src[0]);
      const uint16_t *rhs = reinterpret_cast<const uint16_t *>(src[1]);
      for (size_t i = 0; i != size; ++i, ++lhs, ++rhs) {
        if (*lhs != *rhs) {
          *reinterpret_cast<int *>(dst) = false;
          return;
        }
      }
      *reinterpret_cast<int *>(dst) = true;
    }

    static intptr_t instantiate(
        const char *DYND_UNUSED(static_data), size_t DYND_UNUSED(data_size),
        char *DYND_UNUSED(data), void *ckb, intptr_t ckb_offset,
        const ndt::type &DYND_UNUSED(dst_tp),
        const char *DYND_UNUSED(dst_arrmeta), intptr_t DYND_UNUSED(nsrc),
        const ndt::type *src_tp, const char *const *DYND_UNUSED(src_arrmeta),
        kernel_request_t kernreq, const eval::eval_context *DYND_UNUSED(ectx),
        intptr_t DYND_UNUSED(nkwd), const nd::array *DYND_UNUSED(kwds),
        const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars))
    {
      make(ckb, kernreq, ckb_offset,
           src_tp[0].extended<ndt::fixed_string_type>()->get_size());
      return ckb_offset;
    }
  };

  template <>
  struct fixed_string_equal_kernel<
      string_encoding_utf_32> : base_kernel<fixed_string_equal_kernel<string_encoding_utf_32>,
                                            2> {
    size_t size;

    fixed_string_equal_kernel(size_t size) : size(size)
    {
    }

    void single(char *dst, char *const *src)
    {
      const uint32_t *lhs = reinterpret_cast<const uint32_t *>(src[0]);
      const uint32_t *rhs = reinterpret_cast<const uint32_t *>(src[1]);
      for (size_t i = 0; i != size; ++i, ++lhs, ++rhs) {
        if (*lhs != *rhs) {
          *reinterpret_cast<int *>(dst) = false;
          return;
        }
      }
      *reinterpret_cast<int *>(dst) = true;
    }

    static intptr_t instantiate(
        char *DYND_UNUSED(static_data), size_t DYND_UNUSED(data_size),
        char *DYND_UNUSED(data), void *ckb, intptr_t ckb_offset,
        const ndt::type &DYND_UNUSED(dst_tp),
        const char *DYND_UNUSED(dst_arrmeta), intptr_t DYND_UNUSED(nsrc),
        const ndt::type *src_tp, const char *const *DYND_UNUSED(src_arrmeta),
        kernel_request_t kernreq, const eval::eval_context *DYND_UNUSED(ectx),
        intptr_t DYND_UNUSED(nkwd), const nd::array *DYND_UNUSED(kwds),
        const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars))
    {
      make(ckb, kernreq, ckb_offset,
           src_tp[0].extended<ndt::fixed_string_type>()->get_size());
      return ckb_offset;
    }
  };

  template <>
  struct equal_kernel<
      fixed_string_type_id,
      fixed_string_type_id> : base_virtual_kernel<equal_kernel<fixed_string_type_id,
                                                               fixed_string_type_id>> {
    static intptr_t instantiate(
        char *static_data, size_t data_size, char *data, void *ckb,
        intptr_t ckb_offset, const ndt::type &dst_tp, const char *dst_arrmeta,
        intptr_t nsrc, const ndt::type *src_tp, const char *const *src_arrmeta,
        kernel_request_t kernreq, const eval::eval_context *ectx, intptr_t nkwd,
        const nd::array *kwds, const std::map<std::string, ndt::type> &tp_vars)
    {
      switch (src_tp[0].extended<ndt::fixed_string_type>()->get_encoding()) {
      case string_encoding_ascii:
      case string_encoding_utf_8:
        return fixed_string_equal_kernel<string_encoding_utf_8>::instantiate(
            static_data, data_size, data, ckb, ckb_offset, dst_tp, dst_arrmeta,
            nsrc, src_tp, src_arrmeta, kernreq, ectx, nkwd, kwds, tp_vars);
      case string_encoding_utf_16:
        return fixed_string_equal_kernel<string_encoding_utf_16>::instantiate(
            static_data, data_size, data, ckb, ckb_offset, dst_tp, dst_arrmeta,
            nsrc, src_tp, src_arrmeta, kernreq, ectx, nkwd, kwds, tp_vars);
      case string_encoding_utf_32:
        return fixed_string_equal_kernel<string_encoding_utf_32>::instantiate(
            static_data, data_size, data, ckb, ckb_offset, dst_tp, dst_arrmeta,
            nsrc, src_tp, src_arrmeta, kernreq, ectx, nkwd, kwds, tp_vars);
      default:
        throw std::runtime_error("unidentified string encoding");
      }
    }
  };

  template <>
  struct equal_kernel<
      tuple_type_id,
      tuple_type_id> : base_comparison_kernel<equal_kernel<tuple_type_id,
                                                           tuple_type_id>> {
    typedef equal_kernel extra_type;

    size_t field_count;
    const size_t *src0_data_offsets, *src1_data_offsets;
    // After this are field_count sorting_less kernel offsets, for
    // src0.field_i <op> src1.field_i
    // with each 0 <= i < field_count

    equal_kernel(size_t field_count, const size_t *src0_data_offsets,
                 const size_t *src1_data_offsets)
        : field_count(field_count), src0_data_offsets(src0_data_offsets),
          src1_data_offsets(src1_data_offsets)
    {
    }

    void single(char *dst, char *const *src)
    {
      const size_t *kernel_offsets = reinterpret_cast<const size_t *>(this + 1);
      char *child_src[2];
      for (size_t i = 0; i != field_count; ++i) {
        ckernel_prefix *echild = reinterpret_cast<ckernel_prefix *>(
            reinterpret_cast<char *>(this) + kernel_offsets[i]);
        expr_single_t opchild = echild->get_function<expr_single_t>();
        // if (src0.field_i < src1.field_i) return true
        child_src[0] = src[0] + src0_data_offsets[i];
        child_src[1] = src[1] + src1_data_offsets[i];
        int child_dst;
        opchild(echild, reinterpret_cast<char *>(&child_dst), child_src);
        if (!child_dst) {
          *reinterpret_cast<int *>(dst) = false;
          return;
        }
      }
      *reinterpret_cast<int *>(dst) = true;
    }

    static void destruct(ckernel_prefix *self)
    {
      extra_type *e = reinterpret_cast<extra_type *>(self);
      const size_t *kernel_offsets = reinterpret_cast<const size_t *>(e + 1);
      size_t field_count = e->field_count;
      for (size_t i = 0; i != field_count; ++i) {
        self->get_child_ckernel(kernel_offsets[i])->destroy();
      }
    }

    static intptr_t instantiate(
        char *DYND_UNUSED(static_data), size_t DYND_UNUSED(data_size),
        char *DYND_UNUSED(data), void *ckb, intptr_t ckb_offset,
        const ndt::type &DYND_UNUSED(dst_tp),
        const char *DYND_UNUSED(dst_arrmeta), intptr_t DYND_UNUSED(nsrc),
        const ndt::type *src_tp, const char *const *src_arrmeta,
        kernel_request_t DYND_UNUSED(kernreq), const eval::eval_context *ectx,
        intptr_t DYND_UNUSED(nkwd), const nd::array *DYND_UNUSED(kwds),
        const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars));
  };

  template <type_id_t I0, type_id_t I1>
  struct not_equal_kernel : base_comparison_kernel<not_equal_kernel<I0, I1>> {
    typedef typename type_of<I0>::type A0;
    typedef typename type_of<I1>::type A1;
    typedef typename std::common_type<A0, A1>::type T;

    void single(char *dst, char *const *src)
    {
      *reinterpret_cast<int *>(dst) =
          static_cast<T>(*reinterpret_cast<A0 *>(src[0])) !=
          static_cast<T>(*reinterpret_cast<A1 *>(src[1]));
    }
  };

  template <type_id_t I0>
  struct not_equal_kernel<I0, I0> : base_comparison_kernel<
                                        not_equal_kernel<I0, I0>> {
    typedef typename type_of<I0>::type A0;

    void single(char *dst, char *const *src)
    {
      *reinterpret_cast<int *>(dst) =
          *reinterpret_cast<A0 *>(src[0]) != *reinterpret_cast<A0 *>(src[1]);
    }
  };

  template <string_encoding_t E>
  struct fixed_string_not_equal_kernel;

  template <>
  struct fixed_string_not_equal_kernel<
      string_encoding_utf_8> : base_kernel<fixed_string_not_equal_kernel<string_encoding_utf_8>,
                                           2> {
    size_t size;

    fixed_string_not_equal_kernel(size_t size) : size(size)
    {
    }

    void single(char *dst, char *const *src)
    {
      *reinterpret_cast<int *>(dst) = strncmp(src[0], src[1], size) != 0;
    }

    static intptr_t instantiate(
        char *DYND_UNUSED(static_data), size_t DYND_UNUSED(data_size),
        char *DYND_UNUSED(data), void *ckb, intptr_t ckb_offset,
        const ndt::type &DYND_UNUSED(dst_tp),
        const char *DYND_UNUSED(dst_arrmeta), intptr_t DYND_UNUSED(nsrc),
        const ndt::type *src_tp, const char *const *DYND_UNUSED(src_arrmeta),
        kernel_request_t kernreq, const eval::eval_context *DYND_UNUSED(ectx),
        intptr_t DYND_UNUSED(nkwd), const nd::array *DYND_UNUSED(kwds),
        const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars))
    {
      make(ckb, kernreq, ckb_offset,
           src_tp[0].extended<ndt::fixed_string_type>()->get_size());
      return ckb_offset;
    }
  };

  template <>
  struct fixed_string_not_equal_kernel<
      string_encoding_utf_16> : base_kernel<fixed_string_not_equal_kernel<string_encoding_utf_16>,
                                            2> {
    size_t size;

    fixed_string_not_equal_kernel(size_t size) : size(size)
    {
    }

    void single(char *dst, char *const *src)
    {
      const uint16_t *lhs = reinterpret_cast<const uint16_t *>(src[0]);
      const uint16_t *rhs = reinterpret_cast<const uint16_t *>(src[1]);
      for (size_t i = 0; i != size; ++i, ++lhs, ++rhs) {
        if (*lhs != *rhs) {
          *reinterpret_cast<int *>(dst) = true;
          return;
        }
      }
      *reinterpret_cast<int *>(dst) = false;
    }

    static intptr_t instantiate(
        char *DYND_UNUSED(static_data), size_t DYND_UNUSED(data_size),
        char *DYND_UNUSED(data), void *ckb, intptr_t ckb_offset,
        const ndt::type &DYND_UNUSED(dst_tp),
        const char *DYND_UNUSED(dst_arrmeta), intptr_t DYND_UNUSED(nsrc),
        const ndt::type *src_tp, const char *const *DYND_UNUSED(src_arrmeta),
        kernel_request_t kernreq, const eval::eval_context *DYND_UNUSED(ectx),
        intptr_t DYND_UNUSED(nkwd), const nd::array *DYND_UNUSED(kwds),
        const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars))
    {
      make(ckb, kernreq, ckb_offset,
           src_tp[0].extended<ndt::fixed_string_type>()->get_size());
      return ckb_offset;
    }
  };

  template <>
  struct fixed_string_not_equal_kernel<
      string_encoding_utf_32> : base_kernel<fixed_string_not_equal_kernel<string_encoding_utf_32>,
                                            2> {
    size_t size;

    fixed_string_not_equal_kernel(size_t size) : size(size)
    {
    }

    void single(char *dst, char *const *src)
    {
      const uint32_t *lhs = reinterpret_cast<const uint32_t *>(src[0]);
      const uint32_t *rhs = reinterpret_cast<const uint32_t *>(src[1]);
      for (size_t i = 0; i != size; ++i, ++lhs, ++rhs) {
        if (*lhs != *rhs) {
          *reinterpret_cast<int *>(dst) = true;
          return;
        }
      }
      *reinterpret_cast<int *>(dst) = false;
    }

    static intptr_t instantiate(
        char *DYND_UNUSED(static_data), size_t DYND_UNUSED(data_size),
        char *DYND_UNUSED(data), void *ckb, intptr_t ckb_offset,
        const ndt::type &DYND_UNUSED(dst_tp),
        const char *DYND_UNUSED(dst_arrmeta), intptr_t DYND_UNUSED(nsrc),
        const ndt::type *src_tp, const char *const *DYND_UNUSED(src_arrmeta),
        kernel_request_t kernreq, const eval::eval_context *DYND_UNUSED(ectx),
        intptr_t DYND_UNUSED(nkwd), const nd::array *DYND_UNUSED(kwds),
        const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars))
    {
      make(ckb, kernreq, ckb_offset,
           src_tp[0].extended<ndt::fixed_string_type>()->get_size());
      return ckb_offset;
    }
  };

  template <>
  struct not_equal_kernel<
      fixed_string_type_id,
      fixed_string_type_id> : base_virtual_kernel<not_equal_kernel<fixed_string_type_id,
                                                                   fixed_string_type_id>> {
    static intptr_t instantiate(
        char *static_data, size_t data_size, char *data, void *ckb,
        intptr_t ckb_offset, const ndt::type &dst_tp, const char *dst_arrmeta,
        intptr_t nsrc, const ndt::type *src_tp, const char *const *src_arrmeta,
        kernel_request_t kernreq, const eval::eval_context *ectx, intptr_t nkwd,
        const nd::array *kwds, const std::map<std::string, ndt::type> &tp_vars)
    {
      switch (src_tp[0].extended<ndt::fixed_string_type>()->get_encoding()) {
      case string_encoding_ascii:
      case string_encoding_utf_8:
        return fixed_string_not_equal_kernel<
            string_encoding_utf_8>::instantiate(static_data, data_size, data,
                                                ckb, ckb_offset, dst_tp,
                                                dst_arrmeta, nsrc, src_tp,
                                                src_arrmeta, kernreq, ectx,
                                                nkwd, kwds, tp_vars);
      case string_encoding_utf_16:
        return fixed_string_not_equal_kernel<
            string_encoding_utf_16>::instantiate(static_data, data_size, data,
                                                 ckb, ckb_offset, dst_tp,
                                                 dst_arrmeta, nsrc, src_tp,
                                                 src_arrmeta, kernreq, ectx,
                                                 nkwd, kwds, tp_vars);
      case string_encoding_utf_32:
        return fixed_string_not_equal_kernel<
            string_encoding_utf_32>::instantiate(static_data, data_size, data,
                                                 ckb, ckb_offset, dst_tp,
                                                 dst_arrmeta, nsrc, src_tp,
                                                 src_arrmeta, kernreq, ectx,
                                                 nkwd, kwds, tp_vars);
      default:
        throw std::runtime_error("unidentified string encoding");
      }
    }
  };

  template <typename T>
  struct string_not_equal_kernel : base_kernel<string_not_equal_kernel<T>, 2> {
    void single(char *dst, char *const *src)
    {
      const string_type_data *da =
          reinterpret_cast<const string_type_data *>(src[0]);
      const string_type_data *db =
          reinterpret_cast<const string_type_data *>(src[1]);
      *reinterpret_cast<int *>(dst) =
          (da->end - da->begin != db->end - db->begin) ||
          memcmp(da->begin, db->begin, da->end - da->begin) != 0;
    }
  };

  template <>
  struct not_equal_kernel<
      string_type_id,
      string_type_id> : base_virtual_kernel<not_equal_kernel<string_type_id,
                                                             string_type_id>> {
    static intptr_t instantiate(
        char *static_data, size_t data_size, char *data, void *ckb,
        intptr_t ckb_offset, const ndt::type &dst_tp, const char *dst_arrmeta,
        intptr_t nsrc, const ndt::type *src_tp, const char *const *src_arrmeta,
        kernel_request_t kernreq, const eval::eval_context *ectx, intptr_t nkwd,
        const nd::array *kwds, const std::map<std::string, ndt::type> &tp_vars)
    {
      switch (src_tp[0].extended<ndt::string_type>()->get_encoding()) {
      case string_encoding_ascii:
      case string_encoding_utf_8:
        return string_not_equal_kernel<uint8_t>::instantiate(
            static_data, data_size, data, ckb, ckb_offset, dst_tp, dst_arrmeta,
            nsrc, src_tp, src_arrmeta, kernreq, ectx, nkwd, kwds, tp_vars);
      case string_encoding_utf_16:
        return string_not_equal_kernel<uint16_t>::instantiate(
            static_data, data_size, data, ckb, ckb_offset, dst_tp, dst_arrmeta,
            nsrc, src_tp, src_arrmeta, kernreq, ectx, nkwd, kwds, tp_vars);
      case string_encoding_utf_32:
        return string_not_equal_kernel<uint32_t>::instantiate(
            static_data, data_size, data, ckb, ckb_offset, dst_tp, dst_arrmeta,
            nsrc, src_tp, src_arrmeta, kernreq, ectx, nkwd, kwds, tp_vars);
      default:
        throw std::runtime_error("unidentified string encoding");
      }
    }
  };

  template <>
  struct not_equal_kernel<
      tuple_type_id,
      tuple_type_id> : base_comparison_kernel<not_equal_kernel<tuple_type_id,
                                                               tuple_type_id>> {
    typedef not_equal_kernel extra_type;

    size_t field_count;
    const size_t *src0_data_offsets, *src1_data_offsets;
    // After this are field_count sorting_less kernel offsets, for
    // src0.field_i <op> src1.field_i
    // with each 0 <= i < field_count

    not_equal_kernel(size_t field_count, const size_t *src0_data_offsets,
                     const size_t *src1_data_offsets)
        : field_count(field_count), src0_data_offsets(src0_data_offsets),
          src1_data_offsets(src1_data_offsets)
    {
    }

    void single(char *dst, char *const *src)
    {
      const size_t *kernel_offsets = reinterpret_cast<const size_t *>(this + 1);
      char *child_src[2];
      for (size_t i = 0; i != field_count; ++i) {
        ckernel_prefix *echild = reinterpret_cast<ckernel_prefix *>(
            reinterpret_cast<char *>(this) + kernel_offsets[i]);
        expr_single_t opchild = echild->get_function<expr_single_t>();
        // if (src0.field_i < src1.field_i) return true
        child_src[0] = src[0] + src0_data_offsets[i];
        child_src[1] = src[1] + src1_data_offsets[i];
        int child_dst;
        opchild(echild, reinterpret_cast<char *>(&child_dst), child_src);
        if (child_dst) {
          *reinterpret_cast<int *>(dst) = true;
          return;
        }
      }
      *reinterpret_cast<int *>(dst) = false;
    }

    static void destruct(ckernel_prefix *self)
    {
      extra_type *e = reinterpret_cast<extra_type *>(self);
      const size_t *kernel_offsets = reinterpret_cast<const size_t *>(e + 1);
      size_t field_count = e->field_count;
      for (size_t i = 0; i != field_count; ++i) {
        self->get_child_ckernel(kernel_offsets[i])->destroy();
      }
    }

    static intptr_t instantiate(
        char *static_data, size_t data_size, char *DYND_UNUSED(data), void *ckb,
        intptr_t ckb_offset, const ndt::type &DYND_UNUSED(dst_tp),
        const char *DYND_UNUSED(dst_arrmeta), intptr_t DYND_UNUSED(nsrc),
        const ndt::type *src_tp, const char *const *src_arrmeta,
        kernel_request_t DYND_UNUSED(kernreq), const eval::eval_context *ectx,
        intptr_t DYND_UNUSED(nkwd), const nd::array *DYND_UNUSED(kwds),
        const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars));
  };

  template <type_id_t I0, type_id_t I1>
  struct greater_equal_kernel
      : base_comparison_kernel<greater_equal_kernel<I0, I1>> {
    typedef typename type_of<I0>::type A0;
    typedef typename type_of<I1>::type A1;
    typedef typename std::common_type<A0, A1>::type T;

    void single(char *dst, char *const *src)
    {
      *reinterpret_cast<int *>(dst) =
          static_cast<T>(*reinterpret_cast<A0 *>(src[0])) >=
          static_cast<T>(*reinterpret_cast<A1 *>(src[1]));
    }
  };

  template <type_id_t I0>
  struct greater_equal_kernel<I0, I0> : base_comparison_kernel<
                                            greater_equal_kernel<I0, I0>> {
    typedef typename type_of<I0>::type A0;

    void single(char *dst, char *const *src)
    {
      *reinterpret_cast<int *>(dst) =
          *reinterpret_cast<A0 *>(src[0]) >= *reinterpret_cast<A0 *>(src[1]);
    }
  };

  template <typename T>
  struct string_greater_equal_kernel
      : base_kernel<string_greater_equal_kernel<T>, 2> {
    void single(char *dst, char *const *src)
    {
      const string_type_data *da =
          reinterpret_cast<const string_type_data *>(src[0]);
      const string_type_data *db =
          reinterpret_cast<const string_type_data *>(src[1]);
      *reinterpret_cast<int *>(dst) =
          !std::lexicographical_compare(reinterpret_cast<const T *>(da->begin),
                                        reinterpret_cast<const T *>(da->end),
                                        reinterpret_cast<const T *>(db->begin),
                                        reinterpret_cast<const T *>(db->end));
    }
  };

  template <>
  struct greater_equal_kernel<
      string_type_id,
      string_type_id> : base_virtual_kernel<greater_equal_kernel<string_type_id,
                                                                 string_type_id>> {
    static intptr_t instantiate(
        char *static_data, size_t data_size, char *data, void *ckb,
        intptr_t ckb_offset, const ndt::type &dst_tp, const char *dst_arrmeta,
        intptr_t nsrc, const ndt::type *src_tp, const char *const *src_arrmeta,
        kernel_request_t kernreq, const eval::eval_context *ectx, intptr_t nkwd,
        const nd::array *kwds, const std::map<std::string, ndt::type> &tp_vars)
    {
      switch (src_tp[0].extended<ndt::string_type>()->get_encoding()) {
      case string_encoding_ascii:
      case string_encoding_utf_8:
        return string_greater_equal_kernel<uint8_t>::instantiate(
            static_data, data_size, data, ckb, ckb_offset, dst_tp, dst_arrmeta,
            nsrc, src_tp, src_arrmeta, kernreq, ectx, nkwd, kwds, tp_vars);
      case string_encoding_utf_16:
        return string_greater_equal_kernel<uint16_t>::instantiate(
            static_data, data_size, data, ckb, ckb_offset, dst_tp, dst_arrmeta,
            nsrc, src_tp, src_arrmeta, kernreq, ectx, nkwd, kwds, tp_vars);
      case string_encoding_utf_32:
        return string_greater_equal_kernel<uint32_t>::instantiate(
            static_data, data_size, data, ckb, ckb_offset, dst_tp, dst_arrmeta,
            nsrc, src_tp, src_arrmeta, kernreq, ectx, nkwd, kwds, tp_vars);
      default:
        throw std::runtime_error("unidentified string encoding");
      }
    }
  };

  template <string_encoding_t E>
  struct fixed_string_greater_equal_kernel;

  template <>
  struct fixed_string_greater_equal_kernel<
      string_encoding_utf_8> : base_kernel<fixed_string_greater_equal_kernel<string_encoding_utf_8>,
                                           2> {
    size_t size;

    fixed_string_greater_equal_kernel(size_t size) : size(size)
    {
    }

    void single(char *dst, char *const *src)
    {
      *reinterpret_cast<int *>(dst) = strncmp(src[0], src[1], size) >= 0;
    }

    static intptr_t instantiate(
        char *DYND_UNUSED(static_data), size_t DYND_UNUSED(data_size),
        char *DYND_UNUSED(data), void *ckb, intptr_t ckb_offset,
        const ndt::type &DYND_UNUSED(dst_tp),
        const char *DYND_UNUSED(dst_arrmeta), intptr_t DYND_UNUSED(nsrc),
        const ndt::type *src_tp, const char *const *DYND_UNUSED(src_arrmeta),
        kernel_request_t kernreq, const eval::eval_context *DYND_UNUSED(ectx),
        intptr_t DYND_UNUSED(nkwd), const nd::array *DYND_UNUSED(kwds),
        const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars))
    {
      make(ckb, kernreq, ckb_offset,
           src_tp[0].extended<ndt::fixed_string_type>()->get_size());
      return ckb_offset;
    }
  };

  template <>
  struct fixed_string_greater_equal_kernel<
      string_encoding_utf_16> : base_kernel<fixed_string_greater_equal_kernel<string_encoding_utf_16>,
                                            2> {
    size_t size;

    fixed_string_greater_equal_kernel(size_t size) : size(size)
    {
    }

    void single(char *dst, char *const *src)
    {
      *reinterpret_cast<int *>(dst) =
          !std::lexicographical_compare(
               reinterpret_cast<const uint16_t *>(src[0]),
               reinterpret_cast<const uint16_t *>(src[0]) + size,
               reinterpret_cast<const uint16_t *>(src[1]),
               reinterpret_cast<const uint16_t *>(src[1]) + size);
    }

    static intptr_t instantiate(
        char *DYND_UNUSED(static_data), size_t DYND_UNUSED(data_size),
        char *DYND_UNUSED(data), void *ckb, intptr_t ckb_offset,
        const ndt::type &DYND_UNUSED(dst_tp),
        const char *DYND_UNUSED(dst_arrmeta), intptr_t DYND_UNUSED(nsrc),
        const ndt::type *src_tp, const char *const *DYND_UNUSED(src_arrmeta),
        kernel_request_t kernreq, const eval::eval_context *DYND_UNUSED(ectx),
        intptr_t DYND_UNUSED(nkwd), const nd::array *DYND_UNUSED(kwds),
        const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars))
    {
      make(ckb, kernreq, ckb_offset,
           src_tp[0].extended<ndt::fixed_string_type>()->get_size());
      return ckb_offset;
    }
  };

  template <>
  struct fixed_string_greater_equal_kernel<
      string_encoding_utf_32> : base_kernel<fixed_string_greater_equal_kernel<string_encoding_utf_32>,
                                            2> {
    size_t size;

    fixed_string_greater_equal_kernel(size_t size) : size(size)
    {
    }

    void single(char *dst, char *const *src)
    {
      *reinterpret_cast<int *>(dst) =
          !std::lexicographical_compare(
               reinterpret_cast<const uint32_t *>(src[0]),
               reinterpret_cast<const uint32_t *>(src[0]) + size,
               reinterpret_cast<const uint32_t *>(src[1]),
               reinterpret_cast<const uint32_t *>(src[1]) + size);
    }

    static intptr_t instantiate(
        char *DYND_UNUSED(static_data), size_t DYND_UNUSED(data_size),
        char *DYND_UNUSED(data), void *ckb, intptr_t ckb_offset,
        const ndt::type &DYND_UNUSED(dst_tp),
        const char *DYND_UNUSED(dst_arrmeta), intptr_t DYND_UNUSED(nsrc),
        const ndt::type *src_tp, const char *const *DYND_UNUSED(src_arrmeta),
        kernel_request_t kernreq, const eval::eval_context *DYND_UNUSED(ectx),
        intptr_t DYND_UNUSED(nkwd), const nd::array *DYND_UNUSED(kwds),
        const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars))
    {
      make(ckb, kernreq, ckb_offset,
           src_tp[0].extended<ndt::fixed_string_type>()->get_size());
      return ckb_offset;
    }
  };

  template <>
  struct greater_equal_kernel<
      fixed_string_type_id,
      fixed_string_type_id> : base_virtual_kernel<greater_equal_kernel<fixed_string_type_id,
                                                                       fixed_string_type_id>> {
    static intptr_t instantiate(
        char *static_data, size_t data_size, char *data, void *ckb,
        intptr_t ckb_offset, const ndt::type &dst_tp, const char *dst_arrmeta,
        intptr_t nsrc, const ndt::type *src_tp, const char *const *src_arrmeta,
        kernel_request_t kernreq, const eval::eval_context *ectx, intptr_t nkwd,
        const nd::array *kwds, const std::map<std::string, ndt::type> &tp_vars)
    {
      switch (src_tp[0].extended<ndt::fixed_string_type>()->get_encoding()) {
      case string_encoding_ascii:
      case string_encoding_utf_8:
        return fixed_string_greater_equal_kernel<
            string_encoding_utf_8>::instantiate(static_data, data_size, data,
                                                ckb, ckb_offset, dst_tp,
                                                dst_arrmeta, nsrc, src_tp,
                                                src_arrmeta, kernreq, ectx,
                                                nkwd, kwds, tp_vars);
      case string_encoding_utf_16:
        return fixed_string_greater_equal_kernel<
            string_encoding_utf_16>::instantiate(static_data, data_size, data,
                                                 ckb, ckb_offset, dst_tp,
                                                 dst_arrmeta, nsrc, src_tp,
                                                 src_arrmeta, kernreq, ectx,
                                                 nkwd, kwds, tp_vars);
      case string_encoding_utf_32:
        return fixed_string_greater_equal_kernel<
            string_encoding_utf_32>::instantiate(static_data, data_size, data,
                                                 ckb, ckb_offset, dst_tp,
                                                 dst_arrmeta, nsrc, src_tp,
                                                 src_arrmeta, kernreq, ectx,
                                                 nkwd, kwds, tp_vars);
      default:
        throw std::runtime_error("unidentified string encoding");
      }
    }
  };

  template <type_id_t I0, type_id_t I1>
  struct greater_kernel : base_comparison_kernel<greater_kernel<I0, I1>> {
    typedef typename type_of<I0>::type A0;
    typedef typename type_of<I1>::type A1;
    typedef typename std::common_type<A0, A1>::type T;

    void single(char *dst, char *const *src)
    {
      *reinterpret_cast<int *>(dst) =
          static_cast<T>(*reinterpret_cast<A0 *>(src[0])) >
          static_cast<T>(*reinterpret_cast<A1 *>(src[1]));
    }
  };

  template <type_id_t I0>
  struct greater_kernel<I0,
                        I0> : base_comparison_kernel<greater_kernel<I0, I0>> {
    typedef typename type_of<I0>::type A0;

    void single(char *dst, char *const *src)
    {
      *reinterpret_cast<int *>(dst) =
          *reinterpret_cast<A0 *>(src[0]) > *reinterpret_cast<A0 *>(src[1]);
    }
  };

  template <typename T>
  struct string_greater_kernel : base_kernel<string_greater_kernel<T>, 2> {
    void single(char *dst, char *const *src)
    {
      const string_type_data *da =
          reinterpret_cast<const string_type_data *>(src[0]);
      const string_type_data *db =
          reinterpret_cast<const string_type_data *>(src[1]);
      *reinterpret_cast<int *>(dst) =
          std::lexicographical_compare(reinterpret_cast<const T *>(db->begin),
                                       reinterpret_cast<const T *>(db->end),
                                       reinterpret_cast<const T *>(da->begin),
                                       reinterpret_cast<const T *>(da->end));
    }
  };

  template <>
  struct greater_kernel<
      string_type_id,
      string_type_id> : base_virtual_kernel<greater_kernel<string_type_id,
                                                           string_type_id>> {
    static intptr_t instantiate(
        char *static_data, size_t data_size, char *data, void *ckb,
        intptr_t ckb_offset, const ndt::type &dst_tp, const char *dst_arrmeta,
        intptr_t nsrc, const ndt::type *src_tp, const char *const *src_arrmeta,
        kernel_request_t kernreq, const eval::eval_context *ectx, intptr_t nkwd,
        const nd::array *kwds, const std::map<std::string, ndt::type> &tp_vars)
    {
      switch (src_tp[0].extended<ndt::string_type>()->get_encoding()) {
      case string_encoding_ascii:
      case string_encoding_utf_8:
        return string_greater_kernel<uint8_t>::instantiate(
            static_data, data_size, data, ckb, ckb_offset, dst_tp, dst_arrmeta,
            nsrc, src_tp, src_arrmeta, kernreq, ectx, nkwd, kwds, tp_vars);
      case string_encoding_utf_16:
        return string_greater_kernel<uint16_t>::instantiate(
            static_data, data_size, data, ckb, ckb_offset, dst_tp, dst_arrmeta,
            nsrc, src_tp, src_arrmeta, kernreq, ectx, nkwd, kwds, tp_vars);
      case string_encoding_utf_32:
        return string_greater_kernel<uint32_t>::instantiate(
            static_data, data_size, data, ckb, ckb_offset, dst_tp, dst_arrmeta,
            nsrc, src_tp, src_arrmeta, kernreq, ectx, nkwd, kwds, tp_vars);
      default:
        throw std::runtime_error("unidentified string encoding");
      }
    }
  };

  template <string_encoding_t E>
  struct fixed_string_greater_kernel;

  template <>
  struct fixed_string_greater_kernel<
      string_encoding_utf_8> : base_kernel<fixed_string_greater_kernel<string_encoding_utf_8>,
                                           2> {
    size_t size;

    fixed_string_greater_kernel(size_t size) : size(size)
    {
    }

    void single(char *dst, char *const *src)
    {
      *reinterpret_cast<int *>(dst) = strncmp(src[0], src[1], size) > 0;
    }

    static intptr_t instantiate(
        char *DYND_UNUSED(static_data), size_t DYND_UNUSED(data_size),
        char *DYND_UNUSED(data), void *ckb, intptr_t ckb_offset,
        const ndt::type &DYND_UNUSED(dst_tp),
        const char *DYND_UNUSED(dst_arrmeta), intptr_t DYND_UNUSED(nsrc),
        const ndt::type *src_tp, const char *const *DYND_UNUSED(src_arrmeta),
        kernel_request_t kernreq, const eval::eval_context *DYND_UNUSED(ectx),
        intptr_t DYND_UNUSED(nkwd), const nd::array *DYND_UNUSED(kwds),
        const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars))
    {
      make(ckb, kernreq, ckb_offset,
           src_tp[0].extended<ndt::fixed_string_type>()->get_size());
      return ckb_offset;
    }
  };

  template <>
  struct fixed_string_greater_kernel<
      string_encoding_utf_16> : base_kernel<fixed_string_greater_kernel<string_encoding_utf_16>,
                                            2> {
    size_t size;

    fixed_string_greater_kernel(size_t size) : size(size)
    {
    }

    void single(char *dst, char *const *src)
    {
      *reinterpret_cast<int *>(dst) = std::lexicographical_compare(
          reinterpret_cast<const uint16_t *>(src[1]),
          reinterpret_cast<const uint16_t *>(src[1]) + size,
          reinterpret_cast<const uint16_t *>(src[0]),
          reinterpret_cast<const uint16_t *>(src[0]) + size);
    }

    static intptr_t instantiate(
        char *DYND_UNUSED(static_data), size_t DYND_UNUSED(data_size),
        char *DYND_UNUSED(data), void *ckb, intptr_t ckb_offset,
        const ndt::type &DYND_UNUSED(dst_tp),
        const char *DYND_UNUSED(dst_arrmeta), intptr_t DYND_UNUSED(nsrc),
        const ndt::type *src_tp, const char *const *DYND_UNUSED(src_arrmeta),
        kernel_request_t kernreq, const eval::eval_context *DYND_UNUSED(ectx),
        intptr_t DYND_UNUSED(nkwd), const nd::array *DYND_UNUSED(kwds),
        const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars))
    {
      make(ckb, kernreq, ckb_offset,
           src_tp[0].extended<ndt::fixed_string_type>()->get_size());
      return ckb_offset;
    }
  };

  template <>
  struct fixed_string_greater_kernel<
      string_encoding_utf_32> : base_kernel<fixed_string_greater_kernel<string_encoding_utf_32>,
                                            2> {
    size_t size;

    fixed_string_greater_kernel(size_t size) : size(size)
    {
    }

    void single(char *dst, char *const *src)
    {
      *reinterpret_cast<int *>(dst) = std::lexicographical_compare(
          reinterpret_cast<const uint32_t *>(src[1]),
          reinterpret_cast<const uint32_t *>(src[1]) + size,
          reinterpret_cast<const uint32_t *>(src[0]),
          reinterpret_cast<const uint32_t *>(src[0]) + size);
    }

    static intptr_t instantiate(
        char *DYND_UNUSED(static_data), size_t DYND_UNUSED(data_size),
        char *DYND_UNUSED(data), void *ckb, intptr_t ckb_offset,
        const ndt::type &DYND_UNUSED(dst_tp),
        const char *DYND_UNUSED(dst_arrmeta), intptr_t DYND_UNUSED(nsrc),
        const ndt::type *src_tp, const char *const *DYND_UNUSED(src_arrmeta),
        kernel_request_t kernreq, const eval::eval_context *DYND_UNUSED(ectx),
        intptr_t DYND_UNUSED(nkwd), const nd::array *DYND_UNUSED(kwds),
        const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars))
    {
      make(ckb, kernreq, ckb_offset,
           src_tp[0].extended<ndt::fixed_string_type>()->get_size());
      return ckb_offset;
    }
  };

  template <>
  struct greater_kernel<
      fixed_string_type_id,
      fixed_string_type_id> : base_virtual_kernel<greater_kernel<fixed_string_type_id,
                                                                 fixed_string_type_id>> {
    static intptr_t instantiate(
        char *static_data, size_t data_size, char *data, void *ckb,
        intptr_t ckb_offset, const ndt::type &dst_tp, const char *dst_arrmeta,
        intptr_t nsrc, const ndt::type *src_tp, const char *const *src_arrmeta,
        kernel_request_t kernreq, const eval::eval_context *ectx, intptr_t nkwd,
        const nd::array *kwds, const std::map<std::string, ndt::type> &tp_vars)
    {
      switch (src_tp[0].extended<ndt::fixed_string_type>()->get_encoding()) {
      case string_encoding_ascii:
      case string_encoding_utf_8:
        return fixed_string_greater_kernel<string_encoding_utf_8>::instantiate(
            static_data, data_size, data, ckb, ckb_offset, dst_tp, dst_arrmeta,
            nsrc, src_tp, src_arrmeta, kernreq, ectx, nkwd, kwds, tp_vars);
      case string_encoding_utf_16:
        return fixed_string_greater_kernel<string_encoding_utf_16>::instantiate(
            static_data, data_size, data, ckb, ckb_offset, dst_tp, dst_arrmeta,
            nsrc, src_tp, src_arrmeta, kernreq, ectx, nkwd, kwds, tp_vars);
      case string_encoding_utf_32:
        return fixed_string_greater_kernel<string_encoding_utf_32>::instantiate(
            static_data, data_size, data, ckb, ckb_offset, dst_tp, dst_arrmeta,
            nsrc, src_tp, src_arrmeta, kernreq, ectx, nkwd, kwds, tp_vars);
      default:
        throw std::runtime_error("unidentified string encoding");
      }
    }
  };

} // namespace dynd::nd

namespace ndt {

  template <type_id_t Src0TypeID, type_id_t Src1TypeID>
  struct type::equivalent<nd::less_kernel<Src0TypeID, Src1TypeID>> {
    static type make()
    {
      return callable_type::make(
          ndt::type::make<int>(),
          {ndt::type(Src0TypeID), ndt::type(Src1TypeID)});
    }
  };

  template <type_id_t Src0TypeID, type_id_t Src1TypeID>
  struct type::equivalent<nd::less_equal_kernel<Src0TypeID, Src1TypeID>> {
    static type make()
    {
      return callable_type::make(
          ndt::type::make<int>(),
          {ndt::type(Src0TypeID), ndt::type(Src1TypeID)});
    }
  };

  template <type_id_t Src0TypeID, type_id_t Src1TypeID>
  struct type::equivalent<nd::equal_kernel<Src0TypeID, Src1TypeID>> {
    static type make()
    {
      return callable_type::make(
          ndt::type::make<int>(),
          {ndt::type(Src0TypeID), ndt::type(Src1TypeID)});
    }
  };

  template <type_id_t Src0TypeID, type_id_t Src1TypeID>
  struct type::equivalent<nd::not_equal_kernel<Src0TypeID, Src1TypeID>> {
    static type make()
    {
      return callable_type::make(
          ndt::type::make<int>(),
          {ndt::type(Src0TypeID), ndt::type(Src1TypeID)});
    }
  };

  template <type_id_t Src0TypeID, type_id_t Src1TypeID>
  struct type::equivalent<nd::greater_equal_kernel<Src0TypeID, Src1TypeID>> {
    static type make()
    {
      return callable_type::make(
          ndt::type::make<int>(),
          {ndt::type(Src0TypeID), ndt::type(Src1TypeID)});
    }
  };

  template <type_id_t Src0TypeID, type_id_t Src1TypeID>
  struct type::equivalent<nd::greater_kernel<Src0TypeID, Src1TypeID>> {
    static type make()
    {
      return callable_type::make(
          ndt::type::make<int>(),
          {ndt::type(Src0TypeID), ndt::type(Src1TypeID)});
    }
  };

} // namespace dynd::ndt

} // namespace dynd
