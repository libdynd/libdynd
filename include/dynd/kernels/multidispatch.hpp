//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <set>
#include <unordered_map>

#include <dynd/kernels/base_kernel.hpp>
#include <dynd/kernels/base_virtual_kernel.hpp>
#include <dynd/type.hpp>
#include <dynd/func/arrfunc.hpp>

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

    /**
     * Placeholder hard-coded function for determining allowable
     * implicit conversions during dispatch. Allowing conversions based
     * on ``kind`` of the following forms:
     *
     * uint -> uint, where the size is nondecreasing
     * uint -> int, where the size is increasing
     * int -> int, where the size is nondecreasing
     * uint -> real, where the size is increasing
     * int -> real, where the size is increasing
     * real -> real, where the size is nondecreasing
     * real -> complex, where the size of the real component is nondecreasing
     *
     */
    inline bool
    can_implicitly_convert(const ndt::type &src, const ndt::type &dst,
                           std::map<nd::string, ndt::type> &typevars)
    {
      if (src == dst) {
        return true;
      }
      if (src.get_ndim() > 0 || dst.get_ndim() > 0) {
        ndt::type src_dtype, dst_dtype;
        if (src.match(dst, typevars)) {
          return can_implicitly_convert(src.get_dtype(), dst.get_dtype(),
                                        typevars);
        } else {
          return false;
        }
      }

      if (src.get_kind() == uint_kind &&
          (dst.get_kind() == uint_kind || dst.get_kind() == sint_kind ||
           dst.get_kind() == real_kind)) {
        return src.get_data_size() < dst.get_data_size();
      }
      if (src.get_kind() == sint_kind &&
          (dst.get_kind() == sint_kind || dst.get_kind() == real_kind)) {
        return src.get_data_size() < dst.get_data_size();
      }
      if (src.get_kind() == real_kind) {
        if (dst.get_kind() == real_kind) {
          return src.get_data_size() < dst.get_data_size();
        } else if (dst.get_kind() == complex_kind) {
          return src.get_data_size() * 2 <= dst.get_data_size();
        }
      }
      return false;
    }

    struct old_multidispatch_ck : base_virtual_kernel<old_multidispatch_ck> {
      static intptr_t instantiate(
          const arrfunc_type_data *af_self, const ndt::arrfunc_type *af_tp,
          char *data, void *ckb, intptr_t ckb_offset, const ndt::type &dst_tp,
          const char *dst_arrmeta, intptr_t nsrc, const ndt::type *src_tp,
          const char *const *src_arrmeta, kernel_request_t kernreq,
          const eval::eval_context *ectx, const nd::array &kwds,
          const std::map<dynd::nd::string, ndt::type> &tp_vars);

      static void
      resolve_dst_type(const arrfunc_type_data *af_self,
                       const ndt::arrfunc_type *af_tp, char *data,
                       ndt::type &dst_tp, intptr_t nsrc,
                       const ndt::type *src_tp, const nd::array &kwds,
                       const std::map<nd::string, ndt::type> &tp_vars);
    };

    struct multidispatch_ck : base_virtual_kernel<multidispatch_ck> {
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

      static void resolve_option_values(
          const arrfunc_type_data *self, const ndt::arrfunc_type *af_tp,
          char *DYND_UNUSED(data), intptr_t nsrc, const ndt::type *src_tp,
          nd::array &kwds, const std::map<nd::string, ndt::type> &tp_vars)
      {
        const arrfunc_type_data *child = find(self, tp_vars);
        if (child->resolve_option_values != NULL) {
          child->resolve_option_values(child, af_tp, NULL, nsrc, src_tp, kwds,
                                       tp_vars);
        }
      }

      static void resolve_dst_type(const arrfunc_type_data *self,
                                   const ndt::arrfunc_type *af_tp,
                                   char *DYND_UNUSED(data), ndt::type &dst_tp,
                                   intptr_t nsrc, const ndt::type *src_tp,
                                   const nd::array &kwds,
                                   const std::map<string, ndt::type> &tp_vars)
      {
        const arrfunc_type_data *child = find(self, tp_vars);
        if (child->resolve_dst_type != NULL) {
          child->resolve_dst_type(child, af_tp, NULL, dst_tp, nsrc, src_tp,
                                  kwds, tp_vars);
        } else {
          dst_tp = af_tp->get_return_type();
        }

        dst_tp = ndt::substitute(dst_tp, tp_vars, true);
      }

      static intptr_t
      instantiate(const arrfunc_type_data *self,
                  const ndt::arrfunc_type *self_tp, char *DYND_UNUSED(data),
                  void *ckb, intptr_t ckb_offset, const ndt::type &dst_tp,
                  const char *dst_arrmeta, intptr_t nsrc,
                  const ndt::type *src_tp, const char *const *src_arrmeta,
                  kernel_request_t kernreq, const eval::eval_context *ectx,
                  const array &kwds, const std::map<string, ndt::type> &tp_vars)
      {
        const arrfunc_type_data *child = find(self, tp_vars);
        return child->instantiate(child, self_tp, NULL, ckb, ckb_offset, dst_tp,
                                  dst_arrmeta, nsrc, src_tp, src_arrmeta,
                                  kernreq, ectx, kwds, tp_vars);
      }
    };

    namespace detail {
      template <bool own_children, int N>
      struct multidispatch_by_type_id_kernel_static_data;

      template <>
      struct multidispatch_by_type_id_kernel_static_data<true, 1> {
        arrfunc children[DYND_TYPE_ID_MAX + 1];
        arrfunc default_child;
        intptr_t i0;

        multidispatch_by_type_id_kernel_static_data(
            intptr_t size, const arrfunc *children,
            const arrfunc &default_child, intptr_t i0)
            : default_child(default_child), i0(i0)
        {
          for (intptr_t j = 0; j < size; ++j) {
            const arrfunc &child = children[j];
            if (!child.is_null()) {
              const ndt::type &tp = child.get_type()->get_pos_type(i0);
              this->children[tp.get_type_id()] = child;
            }
          }
        }
      };

      template <>
      struct multidispatch_by_type_id_kernel_static_data<false, 1> {
        const arrfunc *children;
        const arrfunc &default_child;
        intptr_t i0;

        multidispatch_by_type_id_kernel_static_data(
            intptr_t DYND_UNUSED(size), const arrfunc *children,
            const arrfunc &default_child, intptr_t i0)
            : children(children), default_child(default_child), i0(i0)
        {
        }
      };
    }

    template <bool own_children, int N>
    struct multidispatch_by_type_id_kernel;

    template <bool own_children>
    struct multidispatch_by_type_id_kernel<own_children, 1>
        : base_virtual_kernel<
              multidispatch_by_type_id_kernel<own_children, 1>> {

      typedef detail::multidispatch_by_type_id_kernel_static_data<
          own_children, 1> static_data;

      static void
      resolve_dst_type(const arrfunc_type_data *self,
                       const ndt::arrfunc_type *DYND_UNUSED(self_tp),
                       char *DYND_UNUSED(data), ndt::type &dst_tp,
                       intptr_t nsrc, const ndt::type *src_tp,
                       const dynd::nd::array &kwds,
                       const std::map<dynd::nd::string, ndt::type> &tp_vars)
      {
        const std::shared_ptr<static_data> &data =
            *self->get_data_as<std::shared_ptr<static_data>>();
        const ndt::type &tp = (data->i0 == -1) ? dst_tp : src_tp[data->i0];
        arrfunc child = data->children[tp.get_type_id()];
        if (child.is_null()) {
          child = data->default_child;
        }

        if (dst_tp.is_null()) {
          const ndt::type &child_dst_tp = child.get_type()->get_return_type();
          if (child_dst_tp.is_symbolic()) {
            child.get()->resolve_dst_type(child.get(), child.get_type(), NULL,
                                          dst_tp, nsrc, src_tp, kwds, tp_vars);
          } else {
            dst_tp = child_dst_tp;
          }
        }

        //                Uncomment this for the unary arithmetic operators

        //            child.get()->resolve_dst_type(child.get(),
        //            child.get_type(),
        //         NULL, dst_tp, nsrc, src_tp,
        //                                              kwds, tp_vars);
      }

      static intptr_t
      instantiate(const arrfunc_type_data *self,
                  const ndt::arrfunc_type *DYND_UNUSED(self_tp),
                  char *DYND_UNUSED(data), void *ckb, intptr_t ckb_offset,
                  const ndt::type &dst_tp, const char *dst_arrmeta,
                  intptr_t nsrc, const ndt::type *src_tp,
                  const char *const *src_arrmeta, kernel_request_t kernreq,
                  const eval::eval_context *ectx, const dynd::nd::array &kwds,
                  const std::map<dynd::nd::string, ndt::type> &tp_vars)
      {
        const std::shared_ptr<static_data> &data =
            *self->get_data_as<std::shared_ptr<static_data>>();
        const ndt::type &tp = (data->i0 == -1) ? dst_tp : src_tp[data->i0];

        arrfunc child = data->children[tp.get_type_id()];
        if (child.is_null()) {
          child = data->default_child;
        }

        return child.get()->instantiate(child.get(), child.get_type(), NULL,
                                        ckb, ckb_offset, dst_tp, dst_arrmeta,
                                        nsrc, src_tp, src_arrmeta, kernreq,
                                        ectx, kwds, tp_vars);
      }
    };

    template <bool own_children>
    struct multidispatch_by_type_id_kernel<own_children, 2>
        : base_virtual_kernel<
              multidispatch_by_type_id_kernel<own_children, 2>> {
      static void resolve_dst_type(
          const arrfunc_type_data *self, const ndt::arrfunc_type *self_tp,
          char *DYND_UNUSED(data), ndt::type &dst_tp, intptr_t nsrc,
          const ndt::type *src_tp, const dynd::nd::array &kwds,
          const std::map<dynd::nd::string, ndt::type> &tp_vars)
      {
        const arrfunc(*data)[builtin_type_id_count] =
            (*self->get_data_as<
                 const std::unique_ptr<nd::arrfunc[][builtin_type_id_count]>>())
                .get();

        const arrfunc &child =
            data[src_tp[0].get_type_id()][src_tp[1].get_type_id()];
        child.get()->resolve_dst_type(self, self_tp, NULL, dst_tp, nsrc, src_tp,
                                      kwds, tp_vars);
      }

      static intptr_t
      instantiate(const arrfunc_type_data *self,
                  const ndt::arrfunc_type *self_tp, char *DYND_UNUSED(data),
                  void *ckb, intptr_t ckb_offset, const ndt::type &dst_tp,
                  const char *dst_arrmeta, intptr_t nsrc,
                  const ndt::type *src_tp, const char *const *src_arrmeta,
                  kernel_request_t kernreq, const eval::eval_context *ectx,
                  const dynd::nd::array &kwds,
                  const std::map<dynd::nd::string, ndt::type> &tp_vars)
      {
        const arrfunc(*data)[builtin_type_id_count] =
            (*self->get_data_as<
                 const std::unique_ptr<nd::arrfunc[][builtin_type_id_count]>>())
                .get();

        const arrfunc &child =
            data[src_tp[0].get_type_id()][src_tp[1].get_type_id()];
        return child.get()->instantiate(
            self, self_tp, NULL, ckb, ckb_offset, dst_tp, dst_arrmeta, nsrc,
            src_tp, src_arrmeta, kernreq, ectx, kwds, tp_vars);
      }
    };

    template <int N0, int N1>
    struct new_multidispatch_by_type_id_kernel
        : base_virtual_kernel<new_multidispatch_by_type_id_kernel<N0, N1>> {
      static void
      resolve_dst_type(const arrfunc_type_data *self,
                       const ndt::arrfunc_type *DYND_UNUSED(self_tp),
                       char *DYND_UNUSED(data), ndt::type &dst_tp,
                       intptr_t nsrc, const ndt::type *src_tp,
                       const dynd::nd::array &kwds,
                       const std::map<dynd::nd::string, ndt::type> &tp_vars)
      {
        const arrfunc(&children)[N0][N1] =
            self->get_data_as<std::reference_wrapper<arrfunc[N0][N1]>>()->get();

        const arrfunc &child =
            children[src_tp[0].get_type_id()][src_tp[1].get_type_id()];
        if (dst_tp.is_null()) {
          const ndt::type &child_dst_tp = child.get_type()->get_return_type();
          if (child_dst_tp.is_symbolic()) {
            child.get()->resolve_dst_type(child.get(), child.get_type(), NULL,
                                          dst_tp, nsrc, src_tp, kwds, tp_vars);
          } else {
            dst_tp = child_dst_tp;
          }
        }
      }

      static intptr_t
      instantiate(const arrfunc_type_data *self,
                  const ndt::arrfunc_type *DYND_UNUSED(self_tp), char *data,
                  void *ckb, intptr_t ckb_offset, const ndt::type &dst_tp,
                  const char *dst_arrmeta, intptr_t nsrc,
                  const ndt::type *src_tp, const char *const *src_arrmeta,
                  kernel_request_t kernreq, const eval::eval_context *ectx,
                  const dynd::nd::array &kwds,
                  const std::map<dynd::nd::string, ndt::type> &tp_vars)
      {
        const arrfunc(&children)[N0][N1] =
            self->get_data_as<std::reference_wrapper<arrfunc[N0][N1]>>()->get();

        const arrfunc &child =
            children[src_tp[0].get_type_id()][src_tp[1].get_type_id()];
        return child.get()->instantiate(child.get(), child.get_type(), data,
                                        ckb, ckb_offset, dst_tp, dst_arrmeta,
                                        nsrc, src_tp, src_arrmeta, kernreq,
                                        ectx, kwds, tp_vars);
      }
    };

  } // namespace dynd::nd::functional
} // namespace dynd::nd
} // namespace dynd