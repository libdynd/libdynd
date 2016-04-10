//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/callables/base_callable.hpp>
#include <dynd/kernels/range_kernel.hpp>

namespace dynd {
namespace nd {

  template <type_id_t RetElementID, type_id_t RetElementBaseID>
  class range_callable : public base_callable {
    typedef typename type_of<RetElementID>::type ret_element_type;

  public:
    range_callable() : base_callable(ndt::type("(stop: Scalar, start: ?Scalar, step: ?Scalar) -> Fixed * Scalar")) {}

    ndt::type resolve(base_callable *DYND_UNUSED(caller), char *DYND_UNUSED(data), call_graph &cg,
                      const ndt::type &DYND_UNUSED(ret_tp), size_t DYND_UNUSED(narg),
                      const ndt::type *DYND_UNUSED(arg_tp), size_t DYND_UNUSED(nkwd), const array *kwds,
                      const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars)) {
      ret_element_type start;
      if (kwds[1].is_na()) {
        start = 0;
      } else {
        start = kwds[1].as<ret_element_type>();
      }

      ret_element_type stop;

      stop = kwds[0].as<ret_element_type>();

      ret_element_type step;
      if (kwds[2].is_na()) {
        step = 1;
      } else {
        step = kwds[2].as<ret_element_type>();
      }

      cg.emplace_back([start, stop, step](kernel_builder &kb, kernel_request_t kernreq, const char *dst_arrmeta,
                                          size_t DYND_UNUSED(nsrc), const char *const *DYND_UNUSED(src_arrmeta)) {
        kb.emplace_back<range_kernel<RetElementID>>(
            kernreq, start, stop, step,
            reinterpret_cast<const ndt::fixed_dim_type::metadata_type *>(dst_arrmeta)->dim_size,
            reinterpret_cast<const ndt::fixed_dim_type::metadata_type *>(dst_arrmeta)->stride);
      });

      if (step > 0) {
        if (stop <= start) {
          return ndt::make_type<ndt::fixed_dim_type>(0, RetElementID);
        } else {
          return ndt::make_type<ndt::fixed_dim_type>(static_cast<size_t>((stop - start + step - 1) / step),
                                                     RetElementID);
        }
      } else if (step < 0) {
        if (stop >= start) {
          return ndt::make_type<ndt::fixed_dim_type>(0, RetElementID);
        } else {
          return ndt::make_type<ndt::fixed_dim_type>(static_cast<size_t>((stop - start + step + 1) / step),
                                                     RetElementID);
        }
      } else {
        throw std::runtime_error("");
      }
    }
  };

  template <type_id_t RetElementID>
  class range_callable<RetElementID, float_kind_id> : public base_callable {
    typedef typename type_of<RetElementID>::type ret_element_type;

  public:
    range_callable() : base_callable(ndt::type("(stop: Scalar, start: ?Scalar, step: ?Scalar) -> Fixed * Scalar")) {}

    ndt::type resolve(base_callable *DYND_UNUSED(caller), char *DYND_UNUSED(data), call_graph &cg,
                      const ndt::type &DYND_UNUSED(ret_tp), size_t DYND_UNUSED(narg),
                      const ndt::type *DYND_UNUSED(arg_tp), size_t DYND_UNUSED(nkwd), const array *kwds,
                      const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars)) {
      ret_element_type start;
      if (kwds[1].is_na()) {
        start = 0;
      } else {
        start = kwds[1].as<ret_element_type>();
      }

      ret_element_type stop = kwds[0].as<ret_element_type>();

      ret_element_type step;
      if (kwds[2].is_na()) {
        step = 1;
      } else {
        step = kwds[2].as<ret_element_type>();
      }

      cg.emplace_back([start, stop, step](kernel_builder &kb, kernel_request_t kernreq, const char *dst_arrmeta,
                                          size_t DYND_UNUSED(nsrc), const char *const *DYND_UNUSED(src_arrmeta)) {
        kb.emplace_back<range_kernel<RetElementID>>(
            kernreq, start, stop, step,
            reinterpret_cast<const ndt::fixed_dim_type::metadata_type *>(dst_arrmeta)->dim_size,
            reinterpret_cast<const ndt::fixed_dim_type::metadata_type *>(dst_arrmeta)->stride);
      });

      size_t size = static_cast<size_t>(floor((stop - start + static_cast<ret_element_type>(0.5) * step) / step));
      return ndt::make_type<ndt::fixed_dim_type>(size, RetElementID);
    }
  };

  class range_dispatch_callable : public base_callable {
  public:
    range_dispatch_callable()
        : base_callable(ndt::type("(stop: Scalar, start: ?Scalar, step: ?Scalar) -> Fixed * Scalar")) {}

    ndt::type resolve(base_callable *DYND_UNUSED(caller), char *DYND_UNUSED(data), call_graph &cg,
                      const ndt::type &ret_tp, size_t narg, const ndt::type *arg_tp, size_t nkwd, const array *kwds,
                      const std::map<std::string, ndt::type> &tp_vars) {
      static callable fint32 = nd::make_callable<range_callable<int32_id, int_kind_id>>();
      static callable fint64 = nd::make_callable<range_callable<int64_id, int_kind_id>>();
      static callable ffloat32 = nd::make_callable<range_callable<float32_id, float_kind_id>>();
      static callable ffloat64 = nd::make_callable<range_callable<float64_id, float_kind_id>>();

      const ndt::type &ret_element_tp = kwds[0].get_type();
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
