//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/callables/base_callable.hpp>
#include <dynd/kernels/range_kernel.hpp>

namespace dynd {
namespace nd {

  template <typename ReturnElementType, typename Enable = void>
  class range_callable;

  template <typename ReturnElementType>
  class range_callable<ReturnElementType, std::enable_if_t<is_signed_integral<ReturnElementType>::value>>
      : public base_callable {
  public:
    range_callable() : base_callable(ndt::type("(start: ?Scalar, stop: ?Scalar, step: ?Scalar) -> Fixed * Scalar")) {}

    ndt::type resolve(base_callable *DYND_UNUSED(caller), char *DYND_UNUSED(data), call_graph &cg,
                      const ndt::type &DYND_UNUSED(ret_tp), size_t DYND_UNUSED(narg),
                      const ndt::type *DYND_UNUSED(arg_tp), size_t DYND_UNUSED(nkwd), const array *kwds,
                      const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars)) {
      ReturnElementType start;
      ReturnElementType stop;
      ReturnElementType step;

      if (kwds[0].is_na()) {
        start = 0;
        stop = kwds[1].as<ReturnElementType>();
        step = kwds[2].as<ReturnElementType>();
      } else {
        if (kwds[1].is_na() && kwds[2].is_na()) {
          start = 0;
          stop = kwds[0].as<ReturnElementType>();
          step = 1;
        } else if (kwds[2].is_na()) {
          start = kwds[0].as<ReturnElementType>();
          stop = kwds[1].as<ReturnElementType>();
          step = 1;
        } else {
          start = kwds[0].as<ReturnElementType>();
          stop = kwds[1].as<ReturnElementType>();
          step = kwds[2].as<ReturnElementType>();
        }
      }

      cg.emplace_back([start, stop, step](kernel_builder &kb, kernel_request_t kernreq, char *DYND_UNUSED(data),
                                          const char *dst_arrmeta, size_t DYND_UNUSED(nsrc),
                                          const char *const *DYND_UNUSED(src_arrmeta)) {
        kb.emplace_back<range_kernel<ReturnElementType>>(
            kernreq, start, stop, step,
            reinterpret_cast<const ndt::fixed_dim_type::metadata_type *>(dst_arrmeta)->dim_size,
            reinterpret_cast<const ndt::fixed_dim_type::metadata_type *>(dst_arrmeta)->stride);
      });

      if (step > 0) {
        if (stop <= start) {
          return ndt::make_type<ndt::fixed_dim_type>(0, ndt::make_type<ReturnElementType>());
        } else {
          return ndt::make_type<ndt::fixed_dim_type>(static_cast<size_t>((stop - start + step - 1) / step),
                                                     ndt::make_type<ReturnElementType>());
        }
      } else if (step < 0) {
        if (stop >= start) {
          return ndt::make_type<ndt::fixed_dim_type>(0, ndt::make_type<ReturnElementType>());
        } else {
          return ndt::make_type<ndt::fixed_dim_type>(static_cast<size_t>((stop - start + step + 1) / step),
                                                     ndt::make_type<ReturnElementType>());
        }
      } else {
        throw std::runtime_error("");
      }
    }
  };

  template <typename ReturnElementType>
  class range_callable<ReturnElementType, std::enable_if_t<is_floating_point<ReturnElementType>::value>>
      : public base_callable {
  public:
    range_callable() : base_callable(ndt::type("(start: ?Scalar, stop: Scalar, step: ?Scalar) -> Fixed * Scalar")) {}

    ndt::type resolve(base_callable *DYND_UNUSED(caller), char *DYND_UNUSED(data), call_graph &cg,
                      const ndt::type &DYND_UNUSED(ret_tp), size_t DYND_UNUSED(narg),
                      const ndt::type *DYND_UNUSED(arg_tp), size_t DYND_UNUSED(nkwd), const array *kwds,
                      const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars)) {
      ReturnElementType start;
      ReturnElementType stop;
      ReturnElementType step;

      if (kwds[0].is_na()) {
        start = 0;
        stop = kwds[1].as<ReturnElementType>();
        step = kwds[2].as<ReturnElementType>();
      } else {
        if (kwds[1].is_na() && kwds[2].is_na()) {
          start = 0;
          stop = kwds[0].as<ReturnElementType>();
          step = 1;
        } else if (kwds[2].is_na()) {
          start = kwds[0].as<ReturnElementType>();
          stop = kwds[1].as<ReturnElementType>();
          step = 1;
        } else {
          start = kwds[0].as<ReturnElementType>();
          stop = kwds[1].as<ReturnElementType>();
          step = kwds[2].as<ReturnElementType>();
        }
      }

      cg.emplace_back([start, stop, step](kernel_builder &kb, kernel_request_t kernreq, char *DYND_UNUSED(data),
                                          const char *ret_metadata, size_t DYND_UNUSED(nsrc),
                                          const char *const *DYND_UNUSED(src_arrmeta)) {
        kb.emplace_back<range_kernel<ReturnElementType>>(
            kernreq, start, stop, step,
            reinterpret_cast<const ndt::fixed_dim_type::metadata_type *>(ret_metadata)->dim_size,
            reinterpret_cast<const ndt::fixed_dim_type::metadata_type *>(ret_metadata)->stride);
      });

      size_t size = static_cast<size_t>(floor((stop - start + static_cast<ReturnElementType>(0.5) * step) / step));
      return ndt::make_type<ndt::fixed_dim_type>(size, ndt::make_type<ReturnElementType>());
    }
  };

  class range_dispatch_callable : public base_callable {
  public:
    range_dispatch_callable()
        : base_callable(ndt::type("(start: ?Scalar, stop: ?Scalar, step: ?Scalar) -> Fixed * Scalar")) {}

    ndt::type resolve(base_callable *DYND_UNUSED(caller), char *DYND_UNUSED(data), call_graph &cg,
                      const ndt::type &ret_tp, size_t narg, const ndt::type *arg_tp, size_t nkwd, const array *kwds,
                      const std::map<std::string, ndt::type> &tp_vars) {
      static callable fint32 = nd::make_callable<range_callable<int32_t>>();
      static callable fint64 = nd::make_callable<range_callable<int64_t>>();
      static callable ffloat32 = nd::make_callable<range_callable<float>>();
      static callable ffloat64 = nd::make_callable<range_callable<double>>();

      const ndt::type &ret_element_tp = (kwds[0].is_na() ? kwds[1] : kwds[0]).get_type();
      switch (ret_element_tp.get_id()) {
      case int32_id:
        return fint32->resolve(this, nullptr, cg, ret_tp, narg, arg_tp, nkwd, kwds, tp_vars);
      case int64_id:
        return fint64->resolve(this, nullptr, cg, ret_tp, narg, arg_tp, nkwd, kwds, tp_vars);
      case float32_id:
        return ffloat32->resolve(this, nullptr, cg, ret_tp, narg, arg_tp, nkwd, kwds, tp_vars);
      case float64_id:
        return ffloat64->resolve(this, nullptr, cg, ret_tp, narg, arg_tp, nkwd, kwds, tp_vars);
      default:
        throw std::runtime_error("unsupported id");
      }
    }
  };

} // namespace dynd::nd
} // namespace dynd
