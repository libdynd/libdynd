//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/callables/base_instantiable_callable.hpp>
#include <dynd/kernels/assignment_kernels.hpp>

namespace dynd {
namespace nd {

  template <type_id_t ResID, type_id_t Arg0ID>
  class assign_callable : public base_callable {
  public:
    assign_callable()
        : base_callable(
              ndt::callable_type::make(ndt::type(ResID), {ndt::type(Arg0ID)}, {"error_mode"},
                                       {ndt::make_type<ndt::option_type>(ndt::make_type<assign_error_mode>())}))
    {
    }

    void instantiate(char *static_data, char *data, kernel_builder *ckb, const ndt::type &dst_tp,
                     const char *dst_arrmeta, intptr_t nsrc, const ndt::type *src_tp, const char *const *src_arrmeta,
                     kernel_request_t kernreq, intptr_t nkwd, const nd::array *kwds,
                     const std::map<std::string, ndt::type> &tp_vars)
    {
      assign_error_mode error_mode = kwds[0].is_na() ? assign_error_default : kwds[0].as<assign_error_mode>();
      switch (error_mode) {
      case assign_error_default:
      case assign_error_nocheck:
        detail::assignment_kernel<ResID, base_id_of<ResID>::value, Arg0ID, base_id_of<Arg0ID>::value,
                                  assign_error_nocheck>::instantiate(static_data, data, ckb, dst_tp, dst_arrmeta, nsrc,
                                                                     src_tp, src_arrmeta, kernreq, nkwd, kwds, tp_vars);
        break;
      case assign_error_overflow:
        detail::assignment_kernel<ResID, base_id_of<ResID>::value, Arg0ID, base_id_of<Arg0ID>::value,
                                  assign_error_overflow>::instantiate(static_data, data, ckb, dst_tp, dst_arrmeta, nsrc,
                                                                      src_tp, src_arrmeta, kernreq, nkwd, kwds,
                                                                      tp_vars);
        break;
      case assign_error_fractional:
        detail::assignment_kernel<ResID, base_id_of<ResID>::value, Arg0ID, base_id_of<Arg0ID>::value,
                                  assign_error_fractional>::instantiate(static_data, data, ckb, dst_tp, dst_arrmeta,
                                                                        nsrc, src_tp, src_arrmeta, kernreq, nkwd, kwds,
                                                                        tp_vars);
        break;
      case assign_error_inexact:
        detail::assignment_kernel<ResID, base_id_of<ResID>::value, Arg0ID, base_id_of<Arg0ID>::value,
                                  assign_error_inexact>::instantiate(static_data, data, ckb, dst_tp, dst_arrmeta, nsrc,
                                                                     src_tp, src_arrmeta, kernreq, nkwd, kwds, tp_vars);
        break;
      default:
        throw std::runtime_error("error");
      }
    }
  };

  template <>
  class assign_callable<fixed_bytes_id, fixed_bytes_id> : public base_callable {
  public:
    assign_callable()
        : base_callable(
              ndt::callable_type::make(ndt::type(fixed_bytes_id), {ndt::type(fixed_bytes_id)}, {"error_mode"},
                                       {ndt::make_type<ndt::option_type>(ndt::make_type<assign_error_mode>())}))
    {
    }

    void instantiate(char *DYND_UNUSED(static_data), char *DYND_UNUSED(data), kernel_builder *DYND_UNUSED(ckb),
                     const ndt::type &DYND_UNUSED(dst_tp), const char *DYND_UNUSED(dst_arrmeta),
                     intptr_t DYND_UNUSED(nsrc), const ndt::type *DYND_UNUSED(src_tp),
                     const char *const *DYND_UNUSED(src_arrmeta), kernel_request_t DYND_UNUSED(kernreq),
                     intptr_t DYND_UNUSED(nkwd), const nd::array *DYND_UNUSED(kwds),
                     const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars))
    {
      throw std::runtime_error("cannot assign to a fixed_bytes type of a different size");
    }
  };

  template <>
  class assign_callable<float64_id, string_id> : public base_callable {
  public:
    assign_callable()
        : base_callable(
              ndt::callable_type::make(ndt::type(float64_id), {ndt::type(string_id)}, {"error_mode"},
                                       {ndt::make_type<ndt::option_type>(ndt::make_type<assign_error_mode>())}))
    {
    }

    void instantiate(char *DYND_UNUSED(static_data), char *DYND_UNUSED(data), kernel_builder *ckb,
                     const ndt::type &DYND_UNUSED(dst_tp), const char *DYND_UNUSED(dst_arrmeta),
                     intptr_t DYND_UNUSED(nsrc), const ndt::type *src_tp, const char *const *src_arrmeta,
                     kernel_request_t kernreq, intptr_t DYND_UNUSED(nkwd), const nd::array *kwds,
                     const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars))
    {
      assign_error_mode error_mode = kwds[0].is_na() ? assign_error_default : kwds[0].as<assign_error_mode>();
      ckb->emplace_back<
          detail::assignment_kernel<float64_id, float_kind_id, string_id, string_kind_id, assign_error_nocheck>>(
          kernreq, src_tp[0], src_arrmeta[0], error_mode);
    }
  };

  template <>
  class assign_callable<fixed_string_id, string_id> : public base_callable {
  public:
    assign_callable()
        : base_callable(
              ndt::callable_type::make(ndt::type(fixed_string_id), {ndt::type(string_id)}, {"error_mode"},
                                       {ndt::make_type<ndt::option_type>(ndt::make_type<assign_error_mode>())}))
    {
    }

    void instantiate(char *DYND_UNUSED(static_data), char *DYND_UNUSED(data), kernel_builder *ckb,
                     const ndt::type &dst_tp, const char *DYND_UNUSED(dst_arrmeta), intptr_t DYND_UNUSED(nsrc),
                     const ndt::type *src_tp, const char *const *DYND_UNUSED(src_arrmeta), kernel_request_t kernreq,
                     intptr_t DYND_UNUSED(nkwd), const nd::array *kwds,
                     const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars))
    {
      assign_error_mode error_mode = kwds[0].is_na() ? assign_error_default : kwds[0].as<assign_error_mode>();

      const ndt::base_string_type *src_fs = src_tp[0].extended<ndt::base_string_type>();
      ckb->emplace_back<
          detail::assignment_kernel<fixed_string_id, string_kind_id, string_id, string_kind_id, assign_error_nocheck>>(
          kernreq, get_next_unicode_codepoint_function(src_fs->get_encoding(), error_mode),
          get_append_unicode_codepoint_function(dst_tp.extended<ndt::fixed_string_type>()->get_encoding(), error_mode),
          dst_tp.get_data_size(), error_mode != assign_error_nocheck);
    }
  };

  template <>
  class assign_callable<fixed_string_id, fixed_string_id> : public base_callable {
  public:
    assign_callable()
        : base_callable(
              ndt::callable_type::make(ndt::type(fixed_string_id), {ndt::type(fixed_string_id)}, {"error_mode"},
                                       {ndt::make_type<ndt::option_type>(ndt::make_type<assign_error_mode>())}))
    {
    }

    void instantiate(char *DYND_UNUSED(static_data), char *DYND_UNUSED(data), kernel_builder *ckb,
                     const ndt::type &dst_tp, const char *DYND_UNUSED(dst_arrmeta), intptr_t DYND_UNUSED(nsrc),
                     const ndt::type *src_tp, const char *const *DYND_UNUSED(src_arrmeta), kernel_request_t kernreq,
                     intptr_t DYND_UNUSED(nkwd), const nd::array *kwds,
                     const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars))
    {
      assign_error_mode error_mode = kwds[0].is_na() ? assign_error_default : kwds[0].as<assign_error_mode>();

      const ndt::fixed_string_type *src_fs = src_tp[0].extended<ndt::fixed_string_type>();
      ckb->emplace_back<detail::assignment_kernel<fixed_string_id, string_kind_id, fixed_string_id, string_kind_id,
                                                  assign_error_nocheck>>(
          kernreq, get_next_unicode_codepoint_function(src_fs->get_encoding(), error_mode),
          get_append_unicode_codepoint_function(dst_tp.extended<ndt::fixed_string_type>()->get_encoding(), error_mode),
          dst_tp.get_data_size(), src_fs->get_data_size(), error_mode != assign_error_nocheck);
    }
  };

  template <>
  class assign_callable<char_id, string_id> : public base_callable {
  public:
    assign_callable()
        : base_callable(
              ndt::callable_type::make(ndt::type(char_id), {ndt::type(string_id)}, {"error_mode"},
                                       {ndt::make_type<ndt::option_type>(ndt::make_type<assign_error_mode>())}))
    {
    }

    void instantiate(char *DYND_UNUSED(static_data), char *DYND_UNUSED(data), kernel_builder *ckb,
                     const ndt::type &dst_tp, const char *DYND_UNUSED(dst_arrmeta), intptr_t DYND_UNUSED(nsrc),
                     const ndt::type *src_tp, const char *const *DYND_UNUSED(src_arrmeta), kernel_request_t kernreq,
                     intptr_t DYND_UNUSED(nkwd), const nd::array *kwds,
                     const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars))
    {
      assign_error_mode error_mode = kwds[0].is_na() ? assign_error_default : kwds[0].as<assign_error_mode>();

      const ndt::base_string_type *src_fs = src_tp[0].extended<ndt::base_string_type>();
      ckb->emplace_back<
          detail::assignment_kernel<fixed_string_id, string_kind_id, string_id, string_kind_id, assign_error_nocheck>>(
          kernreq, get_next_unicode_codepoint_function(src_fs->get_encoding(), error_mode),
          get_append_unicode_codepoint_function(dst_tp.extended<ndt::char_type>()->get_encoding(), error_mode),
          dst_tp.get_data_size(), error_mode != assign_error_nocheck);
    }
  };

  template <>
  class assign_callable<pointer_id, pointer_id> : public base_callable {
  public:
    assign_callable()
        : base_callable(
              ndt::callable_type::make(ndt::type(pointer_id), {ndt::type(pointer_id)}, {"error_mode"},
                                       {ndt::make_type<ndt::option_type>(ndt::make_type<assign_error_mode>())}))
    {
    }

    void instantiate(char *DYND_UNUSED(static_data), char *DYND_UNUSED(data), kernel_builder *ckb,
                     const ndt::type &dst_tp, const char *dst_arrmeta, intptr_t DYND_UNUSED(nsrc),
                     const ndt::type *src_tp, const char *const *src_arrmeta, kernel_request_t kernreq, intptr_t nkwd,
                     const nd::array *kwds, const std::map<std::string, ndt::type> &tp_vars)
    {
      ckb->emplace_back<assignment_kernel<pointer_id, pointer_id>>(kernreq);

      const char *child_src_arrmeta = src_arrmeta[0] + sizeof(pointer_type_arrmeta);
      assign::get()->instantiate(nd::assign::get()->static_data(), NULL, ckb,
                                 dst_tp.extended<ndt::pointer_type>()->get_target_type(), dst_arrmeta, 1,
                                 &src_tp[0].extended<ndt::pointer_type>()->get_target_type(), &child_src_arrmeta,
                                 kernel_request_single, nkwd, kwds, tp_vars);
    }
  };

  template <>
  class assign_callable<option_id, option_id> : public base_callable {
  public:
    assign_callable()
        : base_callable(
              ndt::callable_type::make(ndt::type(option_id), {ndt::type(option_id)}, {"error_mode"},
                                       {ndt::make_type<ndt::option_type>(ndt::make_type<assign_error_mode>())}))
    {
    }

    void instantiate(char *DYND_UNUSED(static_data), char *DYND_UNUSED(data), kernel_builder *ckb,
                     const ndt::type &dst_tp, const char *dst_arrmeta, intptr_t nsrc, const ndt::type *src_tp,
                     const char *const *src_arrmeta, kernel_request_t kernreq, intptr_t nkwd, const nd::array *kwds,
                     const std::map<std::string, ndt::type> &tp_vars)
    {
      intptr_t ckb_offset = ckb->size();
      intptr_t root_ckb_offset = ckb_offset;
      typedef detail::assignment_kernel<option_id, any_kind_id, option_id, any_kind_id, assign_error_nocheck> self_type;
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

  template <>
  class assign_callable<option_id, float_kind_id> : public base_callable {
  public:
    assign_callable()
        : base_callable(
              ndt::callable_type::make(ndt::type(option_id), {ndt::type(float_kind_id)}, {"error_mode"},
                                       {ndt::make_type<ndt::option_type>(ndt::make_type<assign_error_mode>())}))
    {
    }

    void instantiate(char *DYND_UNUSED(static_data), char *DYND_UNUSED(data), kernel_builder *ckb,
                     const ndt::type &dst_tp, const char *dst_arrmeta, intptr_t nsrc, const ndt::type *src_tp,
                     const char *const *src_arrmeta, kernel_request_t kernreq, intptr_t nkwd, const nd::array *kwds,
                     const std::map<std::string, ndt::type> &tp_vars)
    {
      // Deal with some float32 to option[T] conversions where any NaN is
      // interpreted
      // as NA.
      ndt::type src_tp_as_option = ndt::make_type<ndt::option_type>(src_tp[0]);
      callable f = make_callable<assign_callable<option_id, option_id>>();
      f->instantiate(NULL, NULL, ckb, dst_tp, dst_arrmeta, nsrc, &src_tp_as_option, src_arrmeta, kernreq, nkwd, kwds,
                     tp_vars);
    }
  };

  template <>
  class assign_callable<option_id, string_id> : public base_callable {
  public:
    assign_callable()
        : base_callable(
              ndt::callable_type::make(ndt::type(option_id), {ndt::type(string_id)}, {"error_mode"},
                                       {ndt::make_type<ndt::option_type>(ndt::make_type<assign_error_mode>())}))
    {
    }

    void instantiate(char *DYND_UNUSED(static_data), char *DYND_UNUSED(data), kernel_builder *ckb,
                     const ndt::type &dst_tp, const char *dst_arrmeta, intptr_t nsrc, const ndt::type *src_tp,
                     const char *const *src_arrmeta, kernel_request_t kernreq, intptr_t nkwd, const nd::array *kwds,
                     const std::map<std::string, ndt::type> &tp_vars)
    {
      assign_error_mode error_mode = kwds[0].is_na() ? assign_error_default : kwds[0].as<assign_error_mode>();

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
        ckb->emplace_back<detail::string_to_option_bool_ck>(kernreq);
        return;
      case int8_id:
      case int16_id:
      case int32_id:
      case int64_id:
      case int128_id:
      case float16_id:
      case float32_id:
      case float64_id:
        ckb->emplace_back<detail::string_to_option_number_ck>(kernreq, tid, error_mode);
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
      ckb->emplace_back<detail::string_to_option_tp_ck>(kernreq);
      ckb_offset = ckb->size();
      // First child ckernel is the value assignment
      assign::get()->instantiate(assign::get()->static_data(), NULL, ckb,
                                 dst_tp.extended<ndt::option_type>()->get_value_type(), dst_arrmeta, nsrc, src_tp,
                                 src_arrmeta, kernreq | kernel_request_data_only, nkwd, kwds, tp_vars);
      ckb_offset = ckb->size();
      // Re-acquire self because the address may have changed
      detail::string_to_option_tp_ck *self = ckb->get_at<detail::string_to_option_tp_ck>(root_ckb_offset);
      // Second child ckernel is the NA assignment
      self->m_dst_assign_na_offset = ckb_offset - root_ckb_offset;
      nd::callable &assign_na = nd::assign_na::get();
      assign_na.get()->instantiate(assign_na->static_data(), NULL, ckb, dst_tp, dst_arrmeta, nsrc, src_tp, src_arrmeta,
                                   kernreq | kernel_request_data_only, nkwd, kwds, tp_vars);
      ckb_offset = ckb->size();
    }
  };

  template <>
  class assign_callable<tuple_id, tuple_id> : public base_callable {
  public:
    assign_callable()
        : base_callable(
              ndt::callable_type::make(ndt::type(tuple_id), {ndt::type(tuple_id)}, {"error_mode"},
                                       {ndt::make_type<ndt::option_type>(ndt::make_type<assign_error_mode>())}))
    {
    }

    void instantiate(char *DYND_UNUSED(static_data), char *DYND_UNUSED(data), kernel_builder *ckb,
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
  class assign_callable<struct_id, struct_id> : public base_callable {
  public:
    assign_callable()
        : base_callable(
              ndt::callable_type::make(ndt::type(struct_id), {ndt::type(struct_id)}, {"error_mode"},
                                       {ndt::make_type<ndt::option_type>(ndt::make_type<assign_error_mode>())}))
    {
    }

    void instantiate(char *DYND_UNUSED(static_data), char *DYND_UNUSED(data), kernel_builder *ckb,
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

} // namespace dynd::nd
} // namespace dynd
