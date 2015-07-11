//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/func/arrfunc.hpp>
#include <dynd/kernels/base_kernel.hpp>
#include <dynd/kernels/base_virtual_kernel.hpp>

namespace dynd {
namespace nd {
  namespace detail {

    template <typename T>
    struct is_avail_int_ck
        : base_kernel<is_avail_int_ck<T>, kernel_request_host, 1> {
      void single(char *dst, char *const *src)
      {
        *dst = **reinterpret_cast<T *const *>(src) !=
               std::numeric_limits<T>::min();
      }

      void strided(char *dst, intptr_t dst_stride, char *const *src,
                   const intptr_t *src_stride, size_t count)
      {
        char *src0 = src[0];
        intptr_t src0_stride = src_stride[0];
        for (size_t i = 0; i != count; ++i) {
          *dst = *reinterpret_cast<T *>(src0) != std::numeric_limits<T>::min();
          dst += dst_stride;
          src0 += src0_stride;
        }
      }
    };

    template <typename T>
    struct assign_na_int_ck
        : base_kernel<assign_na_int_ck<T>, kernel_request_host, 1> {
      void single(char *dst, char *const *DYND_UNUSED(src))
      {
        *reinterpret_cast<T *>(dst) = std::numeric_limits<T>::min();
      }

      void strided(char *dst, intptr_t dst_stride,
                   char *const *DYND_UNUSED(src),
                   const intptr_t *DYND_UNUSED(src_stride), size_t count)
      {
        for (size_t i = 0; i != count; ++i, dst += dst_stride) {
          *reinterpret_cast<T *>(dst) = std::numeric_limits<T>::min();
        }
      }
    };

  } // namespace dynd::nd::detail

  template <typename T>
  struct is_avail_ck;

  template <>
  struct is_avail_ck<bool1>
      : base_kernel<is_avail_ck<bool1>, kernel_request_host, 1> {
    void single(char *dst, char *const *src);

    void strided(char *dst, intptr_t dst_stride, char *const *src,
                 const intptr_t *src_stride, size_t count);

    static void resolve_dst_type(
        const ndt::arrfunc_type *DYND_UNUSED(af_tp),
        char *DYND_UNUSED(static_data), size_t DYND_UNUSED(data_size),
        char *DYND_UNUSED(data), ndt::type &dst_tp, intptr_t DYND_UNUSED(nsrc),
        const ndt::type *DYND_UNUSED(src_tp),
        const nd::array &DYND_UNUSED(kwds),
        const std::map<nd::string, ndt::type> &DYND_UNUSED(tp_vars))
    {
      dst_tp = ndt::make_type<bool1>();
    }
  };

  template <>
  struct is_avail_ck<int8_t> : detail::is_avail_int_ck<int8_t> {
    static void resolve_dst_type(
        const ndt::arrfunc_type *DYND_UNUSED(af_tp),
        char *DYND_UNUSED(static_data), size_t DYND_UNUSED(data_size),
        char *DYND_UNUSED(data), ndt::type &dst_tp, intptr_t DYND_UNUSED(nsrc),
        const ndt::type *DYND_UNUSED(src_tp),
        const nd::array &DYND_UNUSED(kwds),
        const std::map<nd::string, ndt::type> &DYND_UNUSED(tp_vars))
    {
      dst_tp = ndt::make_type<bool1>();
    }
  };

  template <>
  struct is_avail_ck<int16_t> : detail::is_avail_int_ck<int16_t> {
    static void resolve_dst_type(
        const ndt::arrfunc_type *DYND_UNUSED(af_tp),
        char *DYND_UNUSED(static_data), size_t DYND_UNUSED(data_size),
        char *DYND_UNUSED(data), ndt::type &dst_tp, intptr_t DYND_UNUSED(nsrc),
        const ndt::type *DYND_UNUSED(src_tp),
        const nd::array &DYND_UNUSED(kwds),
        const std::map<nd::string, ndt::type> &DYND_UNUSED(tp_vars))
    {
      dst_tp = ndt::make_type<bool1>();
    }
  };

  template <>
  struct is_avail_ck<int32_t> : detail::is_avail_int_ck<int32_t> {
    static void resolve_dst_type(
        const ndt::arrfunc_type *DYND_UNUSED(af_tp),
        char *DYND_UNUSED(static_data), size_t DYND_UNUSED(data_size),
        char *DYND_UNUSED(data), ndt::type &dst_tp, intptr_t DYND_UNUSED(nsrc),
        const ndt::type *DYND_UNUSED(src_tp),
        const nd::array &DYND_UNUSED(kwds),
        const std::map<nd::string, ndt::type> &DYND_UNUSED(tp_vars))
    {
      dst_tp = ndt::make_type<bool1>();
    }
  };

  template <>
  struct is_avail_ck<int64_t> : detail::is_avail_int_ck<int64_t> {
    static void resolve_dst_type(
        const ndt::arrfunc_type *DYND_UNUSED(af_tp),
        char *DYND_UNUSED(static_data), size_t DYND_UNUSED(data_size),
        char *DYND_UNUSED(data), ndt::type &dst_tp, intptr_t DYND_UNUSED(nsrc),
        const ndt::type *DYND_UNUSED(src_tp),
        const nd::array &DYND_UNUSED(kwds),
        const std::map<nd::string, ndt::type> &DYND_UNUSED(tp_vars))
    {
      dst_tp = ndt::make_type<bool1>();
    }
  };

  template <>
  struct is_avail_ck<int128> : detail::is_avail_int_ck<int128> {
    static void resolve_dst_type(
        const ndt::arrfunc_type *DYND_UNUSED(af_tp),
        char *DYND_UNUSED(static_data), size_t DYND_UNUSED(data_size),
        char *DYND_UNUSED(data), ndt::type &dst_tp, intptr_t DYND_UNUSED(nsrc),
        const ndt::type *DYND_UNUSED(src_tp),
        const nd::array &DYND_UNUSED(kwds),
        const std::map<nd::string, ndt::type> &DYND_UNUSED(tp_vars))
    {
      dst_tp = ndt::make_type<bool1>();
    }
  };

  template <>
  struct is_avail_ck<float>
      : base_kernel<is_avail_ck<float>, kernel_request_host, 1> {
    void single(char *dst, char *const *src);

    void strided(char *dst, intptr_t dst_stride, char *const *src,
                 const intptr_t *src_stride, size_t count);

    static void resolve_dst_type(
        const ndt::arrfunc_type *DYND_UNUSED(af_tp),
        char *DYND_UNUSED(static_data), size_t DYND_UNUSED(data_size),
        char *DYND_UNUSED(data), ndt::type &dst_tp, intptr_t DYND_UNUSED(nsrc),
        const ndt::type *DYND_UNUSED(src_tp),
        const nd::array &DYND_UNUSED(kwds),
        const std::map<nd::string, ndt::type> &DYND_UNUSED(tp_vars))
    {
      dst_tp = ndt::make_type<bool1>();
    }
  };

  template <>
  struct is_avail_ck<double>
      : base_kernel<is_avail_ck<double>, kernel_request_host, 1> {
    void single(char *dst, char *const *src);

    void strided(char *dst, intptr_t dst_stride, char *const *src,
                 const intptr_t *src_stride, size_t count);

    static void resolve_dst_type(
        const ndt::arrfunc_type *DYND_UNUSED(af_tp),
        char *DYND_UNUSED(static_data), size_t DYND_UNUSED(data_size),
        char *DYND_UNUSED(data), ndt::type &dst_tp, intptr_t DYND_UNUSED(nsrc),
        const ndt::type *DYND_UNUSED(src_tp),
        const nd::array &DYND_UNUSED(kwds),
        const std::map<nd::string, ndt::type> &DYND_UNUSED(tp_vars))
    {
      dst_tp = ndt::make_type<bool1>();
    }
  };

  template <>
  struct is_avail_ck<complex<float>>
      : base_kernel<is_avail_ck<complex<float>>, kernel_request_host, 1> {
    void single(char *dst, char *const *src);

    void strided(char *dst, intptr_t dst_stride, char *const *src,
                 const intptr_t *src_stride, size_t count);

    static void resolve_dst_type(
        const ndt::arrfunc_type *DYND_UNUSED(af_tp),
        char *DYND_UNUSED(static_data), size_t DYND_UNUSED(data_size),
        char *DYND_UNUSED(data), ndt::type &dst_tp, intptr_t DYND_UNUSED(nsrc),
        const ndt::type *DYND_UNUSED(src_tp),
        const nd::array &DYND_UNUSED(kwds),
        const std::map<nd::string, ndt::type> &DYND_UNUSED(tp_vars))
    {
      dst_tp = ndt::make_type<bool1>();
    }
  };

  template <>
  struct is_avail_ck<complex<double>>
      : base_kernel<is_avail_ck<complex<double>>, kernel_request_host, 1> {
    void single(char *dst, char *const *src);

    void strided(char *dst, intptr_t dst_stride, char *const *src,
                 const intptr_t *src_stride, size_t count);

    static void resolve_dst_type(
        const ndt::arrfunc_type *DYND_UNUSED(af_tp),
        char *DYND_UNUSED(static_data), size_t DYND_UNUSED(data_size),
        char *DYND_UNUSED(data), ndt::type &dst_tp, intptr_t DYND_UNUSED(nsrc),
        const ndt::type *DYND_UNUSED(src_tp),
        const nd::array &DYND_UNUSED(kwds),
        const std::map<nd::string, ndt::type> &DYND_UNUSED(tp_vars))
    {
      dst_tp = ndt::make_type<bool1>();
    }
  };

  template <>
  struct is_avail_ck<void>
      : base_kernel<is_avail_ck<void>, kernel_request_host, 1> {
    void single(char *dst, char *const *src);

    void strided(char *dst, intptr_t dst_stride, char *const *src,
                 const intptr_t *src_stride, size_t count);

    static void resolve_dst_type(
        const ndt::arrfunc_type *DYND_UNUSED(af_tp),
        char *DYND_UNUSED(static_data), size_t DYND_UNUSED(data_size),
        char *DYND_UNUSED(data), ndt::type &dst_tp, intptr_t DYND_UNUSED(nsrc),
        const ndt::type *DYND_UNUSED(src_tp),
        const nd::array &DYND_UNUSED(kwds),
        const std::map<nd::string, ndt::type> &DYND_UNUSED(tp_vars))
    {
      dst_tp = ndt::make_type<bool1>();
    }
  };

  template <typename T>
  struct is_avail_ck<T *>
      : base_kernel<is_avail_ck<T *>, kernel_request_host, 1> {
    void single(char *DYND_UNUSED(dst), char *const *DYND_UNUSED(src))
    {
      throw std::runtime_error("is_avail for pointers is not yet implemented");
    }

    void strided(char *DYND_UNUSED(dst), intptr_t DYND_UNUSED(dst_stride),
                 char *const *DYND_UNUSED(src),
                 const intptr_t *DYND_UNUSED(src_stride),
                 size_t DYND_UNUSED(count))
    {
      throw std::runtime_error("is_avail for pointers is not yet implemented");
    }

    static void resolve_dst_type(
        const ndt::arrfunc_type *DYND_UNUSED(af_tp),
        char *DYND_UNUSED(static_data), size_t DYND_UNUSED(data_size),
        char *DYND_UNUSED(data), ndt::type &dst_tp, intptr_t DYND_UNUSED(nsrc),
        const ndt::type *DYND_UNUSED(src_tp),
        const nd::array &DYND_UNUSED(kwds),
        const std::map<nd::string, ndt::type> &DYND_UNUSED(tp_vars))
    {
      dst_tp = ndt::make_type<bool1>();
    }
  };

  template <typename T>
  struct assign_na_ck;

  template <>
  struct assign_na_ck<bool1>
      : base_kernel<assign_na_ck<bool1>, kernel_request_host, 1> {
    void single(char *dst, char *const *src);

    void strided(char *dst, intptr_t dst_stride, char *const *src,
                 const intptr_t *src_stride, size_t count);
  };

  template <>
  struct assign_na_ck<int8_t> : detail::assign_na_int_ck<int8_t> {
  };

  template <>
  struct assign_na_ck<int16_t> : detail::assign_na_int_ck<int16_t> {
  };

  template <>
  struct assign_na_ck<int32_t> : detail::assign_na_int_ck<int32_t> {
  };

  template <>
  struct assign_na_ck<int64_t> : detail::assign_na_int_ck<int64_t> {
  };

  template <>
  struct assign_na_ck<int128> : detail::assign_na_int_ck<int128> {
  };

  template <>
  struct assign_na_ck<float>
      : base_kernel<assign_na_ck<float>, kernel_request_host, 1> {
    void single(char *dst, char *const *src);

    void strided(char *dst, intptr_t dst_stride, char *const *src,
                 const intptr_t *src_stride, size_t count);
  };

  template <>
  struct assign_na_ck<double>
      : base_kernel<assign_na_ck<double>, kernel_request_host, 1> {
    void single(char *dst, char *const *src);

    void strided(char *dst, intptr_t dst_stride, char *const *src,
                 const intptr_t *src_stride, size_t count);
  };

  template <>
  struct assign_na_ck<dynd::complex<float>>
      : base_kernel<assign_na_ck<dynd::complex<float>>, kernel_request_host,
                    1> {
    void single(char *dst, char *const *src);

    void strided(char *dst, intptr_t dst_stride, char *const *src,
                 const intptr_t *src_stride, size_t count);
  };

  template <>
  struct assign_na_ck<dynd::complex<double>>
      : base_kernel<assign_na_ck<dynd::complex<double>>, kernel_request_host,
                    1> {
    void single(char *dst, char *const *src);

    void strided(char *dst, intptr_t dst_stride, char *const *src,
                 const intptr_t *src_stride, size_t count);
  };

  template <>
  struct assign_na_ck<void>
      : base_kernel<assign_na_ck<void>, kernel_request_host, 1> {
    void single(char *dst, char *const *src);

    void strided(char *dst, intptr_t dst_stride, char *const *src,
                 const intptr_t *src_stride, size_t count);
  };

  template <typename T>
  struct assign_na_ck<T *>
      : base_kernel<assign_na_ck<T *>, kernel_request_host, 1> {
    void single(char *DYND_UNUSED(dst), char *const *DYND_UNUSED(src))
    {
      throw std::runtime_error("assign_na for pointers is not yet implemented");
    }

    void strided(char *DYND_UNUSED(dst), intptr_t DYND_UNUSED(dst_stride),
                 char *const *DYND_UNUSED(src),
                 const intptr_t *DYND_UNUSED(src_stride),
                 size_t DYND_UNUSED(count))
    {
      throw std::runtime_error("assign_na for pointers is not yet implemented");
    }
  };

} // namespace dynd::nd

namespace kernels {

  struct fixed_dim_is_avail_ck {
    static intptr_t
    instantiate(const arrfunc_type_data *self, const ndt::arrfunc_type *af_tp,
                const char *static_data, size_t data_size, char *data,
                void *ckb, intptr_t ckb_offset, const ndt::type &dst_tp,
                const char *dst_arrmeta, intptr_t nsrc, const ndt::type *src_tp,
                const char *const *src_arrmeta, kernel_request_t kernreq,
                const eval::eval_context *ectx, const nd::array &kwds,
                const std::map<nd::string, ndt::type> &tp_vars);
  };

  struct fixed_dim_assign_na_ck {
    static intptr_t
    instantiate(const arrfunc_type_data *self, const ndt::arrfunc_type *af_tp,
                const char *static_data, size_t data_size, char *data,
                void *ckb, intptr_t ckb_offset, const ndt::type &dst_tp,
                const char *dst_arrmeta, intptr_t nsrc, const ndt::type *src_tp,
                const char *const *src_arrmeta, kernel_request_t kernreq,
                const eval::eval_context *ectx, const nd::array &kwds,
                const std::map<nd::string, ndt::type> &tp_vars);
  };

  /**
   * Returns the nafunc structure for the given builtin type id.
   */
  const nd::array &get_option_builtin_nafunc(type_id_t tid);

  /**
   * Returns the nafunc structure for the given pointer to builtin type id.
   */
  const nd::array &get_option_builtin_pointer_nafunc(type_id_t tid);
}
} // namespace dynd::kernels
