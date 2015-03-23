//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <set>
#include <unordered_map>

#include <dynd/kernels/expr_kernels.hpp>
#include <dynd/kernels/virtual.hpp>
#include <dynd/type.hpp>

namespace std {

template <>
struct hash<dynd::ndt::type> {
  size_t operator()(const dynd::ndt::type &tp) const
  {
    return tp.get_type_id();
  }
};
template <>
struct hash<dynd::nd::string> {
  size_t operator()(const dynd::nd::string &) const { return 0; }
};
template <>
struct hash<std::vector<dynd::ndt::type>> {
  size_t operator()(const std::vector<dynd::ndt::type> &v) const
  {
    std::hash<dynd::ndt::type> hash;
    size_t value = 0;
    for (dynd::ndt::type tp : v) {
      value ^= hash(tp) + 0x9e3779b9 + (value << 6) + (value >> 2);
    }
    return value;
  }
};

} // namespace std

namespace dynd {
namespace nd {
  namespace functional {

    struct multidispatch_ck : virtual_ck<multidispatch_ck> {
      typedef std::unordered_map<std::vector<ndt::type>, arrfunc> map_type;

      struct data_type {
        std::shared_ptr<map_type> map;
        std::shared_ptr<std::vector<string>> vars;

        data_type(std::shared_ptr<map_type> map,
                  std::shared_ptr<std::vector<string>> vars)
            : map(map), vars(vars)
        {
        }
      };

      static const arrfunc_type_data *
      find(const arrfunc_type_data *self,
           const std::map<string, ndt::type> &tp_vars)
      {
        const data_type *data = self->get_data_as<data_type>();
        std::shared_ptr<map_type> map = data->map;
        std::shared_ptr<std::vector<string>> vars = data->vars;

        std::vector<ndt::type> tp_vals;
        for (auto pair : tp_vars) {
          if (std::find(vars->begin(), vars->end(), pair.first) !=
              vars->end()) {
            tp_vals.push_back(pair.second);
          }
        }

        return (*map)[tp_vals].get();
      }

      static void
      resolve_option_values(const arrfunc_type_data *self,
                            const arrfunc_type *af_tp, intptr_t nsrc,
                            const ndt::type *src_tp, nd::array &kwds,
                            const std::map<nd::string, ndt::type> &tp_vars)
      {
        const arrfunc_type_data *child = find(self, tp_vars);
        if (child->resolve_option_values != NULL) {
          child->resolve_option_values(child, af_tp, nsrc, src_tp, kwds,
                                       tp_vars);
        }
      }

      static int resolve_dst_type(const arrfunc_type_data *self,
                                  const arrfunc_type *af_tp, intptr_t nsrc,
                                  const ndt::type *src_tp, int throw_on_error,
                                  ndt::type &out_dst_tp, const nd::array &kwds,
                                  const std::map<string, ndt::type> &tp_vars)
      {
        const arrfunc_type_data *child = find(self, tp_vars);
        if (child->resolve_dst_type != NULL) {
          child->resolve_dst_type(child, af_tp, nsrc, src_tp, throw_on_error,
                                  out_dst_tp, kwds, tp_vars);
        } else {
          out_dst_tp = af_tp->get_return_type();
        }

        out_dst_tp = ndt::substitute(out_dst_tp, tp_vars, true);

        return 1;
      }

      static intptr_t
      instantiate(const arrfunc_type_data *self, const arrfunc_type *self_tp,
                  void *ckb, intptr_t ckb_offset, const ndt::type &dst_tp,
                  const char *dst_arrmeta, intptr_t nsrc,
                  const ndt::type *src_tp, const char *const *src_arrmeta,
                  kernel_request_t kernreq, const eval::eval_context *ectx,
                  const array &kwds, const std::map<string, ndt::type> &tp_vars)
      {
        const arrfunc_type_data *child = find(self, tp_vars);
        return child->instantiate(child, self_tp, ckb, ckb_offset, dst_tp,
                                  dst_arrmeta, nsrc, src_tp, src_arrmeta,
                                  kernreq, ectx, kwds, tp_vars);
      }
    };

    template <int N>
    struct multidispatch_by_type_id_ck;

    template <>
    struct multidispatch_by_type_id_ck<1>
        : virtual_ck<multidispatch_by_type_id_ck<1>> {

      static int
      resolve_dst_type(const arrfunc_type_data *self,
                       const arrfunc_type *self_tp, intptr_t nsrc,
                       const ndt::type *src_tp, int throw_on_error,
                       ndt::type &out_dst_tp, const dynd::nd::array &kwds,
                       const std::map<dynd::nd::string, ndt::type> &tp_vars)
      {
        const arrfunc *data = *self->get_data_as<const arrfunc *>();

        const arrfunc &child = data[src_tp[0].get_type_id()];
        return child.get()->resolve_dst_type(self, self_tp, nsrc, src_tp,
                                             throw_on_error, out_dst_tp, kwds,
                                             tp_vars);
      }

      static intptr_t
      instantiate(const arrfunc_type_data *self, const arrfunc_type *self_tp,
                  void *ckb, intptr_t ckb_offset, const ndt::type &dst_tp,
                  const char *dst_arrmeta, intptr_t nsrc,
                  const ndt::type *src_tp, const char *const *src_arrmeta,
                  kernel_request_t kernreq, const eval::eval_context *ectx,
                  const dynd::nd::array &kwds,
                  const std::map<dynd::nd::string, ndt::type> &tp_vars)
      {
        const arrfunc *data = *self->get_data_as<const arrfunc *>();

        const arrfunc &child = data[src_tp[0].get_type_id()];
        return child.get()->instantiate(self, self_tp, ckb, ckb_offset, dst_tp,
                                        dst_arrmeta, nsrc, src_tp, src_arrmeta,
                                        kernreq, ectx, kwds, tp_vars);
      }
    };

    template <>
    struct multidispatch_by_type_id_ck<2>
        : virtual_ck<multidispatch_by_type_id_ck<2>> {
      static int
      resolve_dst_type(const arrfunc_type_data *self,
                       const arrfunc_type *self_tp, intptr_t nsrc,
                       const ndt::type *src_tp, int throw_on_error,
                       ndt::type &out_dst_tp, const dynd::nd::array &kwds,
                       const std::map<dynd::nd::string, ndt::type> &tp_vars)
      {
        const arrfunc(*data)[builtin_type_id_count] =
            *self->get_data_as<const arrfunc(*)[builtin_type_id_count]>();

        const arrfunc &child =
            data[src_tp[0].get_type_id()][src_tp[1].get_type_id()];
        return child.get()->resolve_dst_type(self, self_tp, nsrc, src_tp,
                                             throw_on_error, out_dst_tp, kwds,
                                             tp_vars);
      }

      static intptr_t
      instantiate(const arrfunc_type_data *self, const arrfunc_type *self_tp,
                  void *ckb, intptr_t ckb_offset, const ndt::type &dst_tp,
                  const char *dst_arrmeta, intptr_t nsrc,
                  const ndt::type *src_tp, const char *const *src_arrmeta,
                  kernel_request_t kernreq, const eval::eval_context *ectx,
                  const dynd::nd::array &kwds,
                  const std::map<dynd::nd::string, ndt::type> &tp_vars)
      {
        const arrfunc(*data)[builtin_type_id_count] =
            *self->get_data_as<const arrfunc(*)[builtin_type_id_count]>();

        const arrfunc &child =
            data[src_tp[0].get_type_id()][src_tp[1].get_type_id()];
        return child.get()->instantiate(self, self_tp, ckb, ckb_offset, dst_tp,
                                        dst_arrmeta, nsrc, src_tp, src_arrmeta,
                                        kernreq, ectx, kwds, tp_vars);
      }
    };

  } // namespace dynd::nd::functional
} // namespace dynd::nd
} // namespace dynd