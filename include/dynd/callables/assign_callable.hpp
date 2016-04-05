//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/callables/base_callable.hpp>
#include <dynd/kernels/assignment_kernels.hpp>
#include <dynd/functional.hpp>

namespace dynd {
namespace nd {

  template <type_id_t ResID, type_id_t Arg0ID>
  class assign_callable : public base_callable {
  public:
    assign_callable()
        : base_callable(
              ndt::callable_type::make(ndt::type(ResID), {ndt::type(Arg0ID)}, {"error_mode"},
                                       {ndt::make_type<ndt::option_type>(ndt::make_type<assign_error_mode>())})) {}

    ndt::type resolve(base_callable *DYND_UNUSED(caller), char *DYND_UNUSED(data), call_graph &cg,
                      const ndt::type &dst_tp, size_t DYND_UNUSED(nsrc), const ndt::type *DYND_UNUSED(src_tp),
                      size_t DYND_UNUSED(nkwd), const array *kwds,
                      const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars)) {
      assign_error_mode error_mode =
          (kwds == NULL || kwds[0].is_na()) ? assign_error_default : kwds[0].as<assign_error_mode>();
      switch (error_mode) {
      case assign_error_default:
      case assign_error_nocheck:
        cg.push_back([](kernel_builder *ckb, kernel_request_t kernreq, const char *DYND_UNUSED(dst_arrmeta),
                        size_t DYND_UNUSED(nsrc), const char *const *DYND_UNUSED(src_arrmeta)) {
          ckb->emplace_back<detail::assignment_kernel<ResID, base_id_of<ResID>::value, Arg0ID,
                                                      base_id_of<Arg0ID>::value, assign_error_nocheck>>(kernreq);
        });
        break;
      case assign_error_overflow:
        cg.push_back([](kernel_builder *ckb, kernel_request_t kernreq, const char *DYND_UNUSED(dst_arrmeta),
                        size_t DYND_UNUSED(nsrc), const char *const *DYND_UNUSED(src_arrmeta)) {
          ckb->emplace_back<detail::assignment_kernel<ResID, base_id_of<ResID>::value, Arg0ID,
                                                      base_id_of<Arg0ID>::value, assign_error_overflow>>(kernreq);
        });
        break;
      case assign_error_fractional:
        cg.push_back([](kernel_builder *ckb, kernel_request_t kernreq, const char *DYND_UNUSED(dst_arrmeta),
                        size_t DYND_UNUSED(nsrc), const char *const *DYND_UNUSED(src_arrmeta)) {
          ckb->emplace_back<detail::assignment_kernel<ResID, base_id_of<ResID>::value, Arg0ID,
                                                      base_id_of<Arg0ID>::value, assign_error_fractional>>(kernreq);
        });
        break;
      case assign_error_inexact:
        cg.push_back([](kernel_builder *ckb, kernel_request_t kernreq, const char *DYND_UNUSED(dst_arrmeta),
                        size_t DYND_UNUSED(nsrc), const char *const *DYND_UNUSED(src_arrmeta)) {
          ckb->emplace_back<detail::assignment_kernel<ResID, base_id_of<ResID>::value, Arg0ID,
                                                      base_id_of<Arg0ID>::value, assign_error_inexact>>(kernreq);
        });
        break;
      default:
        throw std::runtime_error("error");
      }

      return dst_tp;
    }
  };

  template <>
  class assign_callable<bool_id, string_id> : public base_callable {
  public:
    assign_callable()
        : base_callable(
              ndt::callable_type::make(ndt::type(bool_id), {ndt::type(string_id)}, {"error_mode"},
                                       {ndt::make_type<ndt::option_type>(ndt::make_type<assign_error_mode>())})) {}

    ndt::type resolve(base_callable *DYND_UNUSED(caller), char *DYND_UNUSED(data), call_graph &cg,
                      const ndt::type &dst_tp, size_t DYND_UNUSED(nsrc), const ndt::type *src_tp,
                      size_t DYND_UNUSED(nkwd), const array *kwds,
                      const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars)) {
      assign_error_mode error_mode = kwds[0].is_na() ? assign_error_default : kwds[0].as<assign_error_mode>();
      switch (error_mode) {
      case assign_error_default:
      case assign_error_nocheck:
        cg.push_back([=](kernel_builder *ckb, kernel_request_t kernreq, const char *DYND_UNUSED(dst_arrmeta),
                         size_t DYND_UNUSED(nsrc), const char *const *src_arrmeta) {
          ckb->emplace_back<
              detail::assignment_kernel<bool_id, bool_kind_id, string_id, string_kind_id, assign_error_nocheck>>(
              kernreq, src_tp[0], src_arrmeta[0]);
        });
        break;
      case assign_error_overflow:
        cg.push_back([=](kernel_builder *ckb, kernel_request_t kernreq, const char *DYND_UNUSED(dst_arrmeta),
                         size_t DYND_UNUSED(nsrc), const char *const *src_arrmeta) {
          ckb->emplace_back<
              detail::assignment_kernel<bool_id, bool_kind_id, string_id, string_kind_id, assign_error_overflow>>(
              kernreq, src_tp[0], src_arrmeta[0]);
        });
        break;
      case assign_error_fractional:
        cg.push_back([=](kernel_builder *ckb, kernel_request_t kernreq, const char *DYND_UNUSED(dst_arrmeta),
                         size_t DYND_UNUSED(nsrc), const char *const *src_arrmeta) {
          ckb->emplace_back<
              detail::assignment_kernel<bool_id, bool_kind_id, string_id, string_kind_id, assign_error_fractional>>(
              kernreq, src_tp[0], src_arrmeta[0]);
        });
        break;
      case assign_error_inexact:
        cg.push_back([=](kernel_builder *ckb, kernel_request_t kernreq, const char *DYND_UNUSED(dst_arrmeta),
                         size_t DYND_UNUSED(nsrc), const char *const *src_arrmeta) {
          ckb->emplace_back<
              detail::assignment_kernel<bool_id, bool_kind_id, string_id, string_kind_id, assign_error_inexact>>(
              kernreq, src_tp[0], src_arrmeta[0]);
        });
        break;
      default:
        throw std::runtime_error("error");
      }

      return dst_tp;
    }

    void instantiate(call_node *&DYND_UNUSED(node), char *DYND_UNUSED(data), kernel_builder *ckb,
                     const ndt::type &DYND_UNUSED(dst_tp), const char *DYND_UNUSED(dst_arrmeta),
                     intptr_t DYND_UNUSED(nsrc), const ndt::type *src_tp, const char *const *src_arrmeta,
                     kernel_request_t kernreq, intptr_t DYND_UNUSED(nkwd), const nd::array *kwds,
                     const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars)) {
      assign_error_mode error_mode = kwds[0].is_na() ? assign_error_default : kwds[0].as<assign_error_mode>();
      switch (error_mode) {
      case assign_error_default:
      case assign_error_nocheck:
        ckb->emplace_back<
            detail::assignment_kernel<bool_id, bool_kind_id, string_id, string_kind_id, assign_error_nocheck>>(
            kernreq, src_tp[0], src_arrmeta[0]);
        break;
      case assign_error_overflow:
        ckb->emplace_back<
            detail::assignment_kernel<bool_id, bool_kind_id, string_id, string_kind_id, assign_error_overflow>>(
            kernreq, src_tp[0], src_arrmeta[0]);
        break;
      case assign_error_fractional:
        ckb->emplace_back<
            detail::assignment_kernel<bool_id, bool_kind_id, string_id, string_kind_id, assign_error_fractional>>(
            kernreq, src_tp[0], src_arrmeta[0]);
        break;
      case assign_error_inexact:
        ckb->emplace_back<
            detail::assignment_kernel<bool_id, bool_kind_id, string_id, string_kind_id, assign_error_inexact>>(
            kernreq, src_tp[0], src_arrmeta[0]);
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
                                       {ndt::make_type<ndt::option_type>(ndt::make_type<assign_error_mode>())})) {}

    ndt::type resolve(base_callable *DYND_UNUSED(caller), char *DYND_UNUSED(data), call_graph &DYND_UNUSED(cg),
                      const ndt::type &dst_tp, size_t DYND_UNUSED(nsrc), const ndt::type *DYND_UNUSED(src_tp),
                      size_t DYND_UNUSED(nkwd), const array *DYND_UNUSED(kwds),
                      const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars)) {
      return dst_tp;
    }

    void instantiate(call_node *&DYND_UNUSED(node), char *DYND_UNUSED(data), kernel_builder *DYND_UNUSED(ckb),
                     const ndt::type &DYND_UNUSED(dst_tp), const char *DYND_UNUSED(dst_arrmeta),
                     intptr_t DYND_UNUSED(nsrc), const ndt::type *DYND_UNUSED(src_tp),
                     const char *const *DYND_UNUSED(src_arrmeta), kernel_request_t DYND_UNUSED(kernreq),
                     intptr_t DYND_UNUSED(nkwd), const nd::array *DYND_UNUSED(kwds),
                     const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars)) {
      throw std::runtime_error("cannot assign to a fixed_bytes type of a different size");
    }
  };

  template <type_id_t IntID>
  class int_to_string_assign_callable : public base_callable {
  public:
    int_to_string_assign_callable()
        : base_callable(
              ndt::callable_type::make(ndt::type(IntID), {ndt::type(string_id)}, {"error_mode"},
                                       {ndt::make_type<ndt::option_type>(ndt::make_type<assign_error_mode>())})) {}

    ndt::type resolve(base_callable *DYND_UNUSED(caller), char *DYND_UNUSED(data), call_graph &cg,
                      const ndt::type &dst_tp, size_t DYND_UNUSED(nsrc), const ndt::type *src_tp,
                      size_t DYND_UNUSED(nkwd), const array *kwds,
                      const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars)) {
      assign_error_mode error_mode = kwds[0].is_na() ? assign_error_default : kwds[0].as<assign_error_mode>();

      type_id_t src0_id = src_tp[0].get_id();
      size_t src0_size = 0;
      string_encoding_t src0_encoding = string_encoding_ascii;
      if (src0_id == fixed_string_id) {
        src0_size = src_tp[0].extended<ndt::fixed_string_type>()->get_size();
        src0_encoding = src_tp[0].extended<ndt::fixed_string_type>()->get_encoding();
      }

      cg.push_back([src0_id, src0_size, src0_encoding,
                    error_mode](kernel_builder *ckb, kernel_request_t kernreq, const char *DYND_UNUSED(dst_arrmeta),
                                size_t DYND_UNUSED(nsrc), const char *const *src_arrmeta) {
        ndt::type string_type = src0_id == fixed_string_id
                                    ? ndt::make_type<ndt::fixed_string_type>(src0_size, src0_encoding)
                                    : ndt::make_type<ndt::string_type>();

        ckb->emplace_back<
            detail::assignment_kernel<IntID, int_kind_id, string_id, string_kind_id, assign_error_default>>(
            kernreq, string_type, src_arrmeta[0], error_mode);
      });

      return dst_tp;
    }
  };

  template <type_id_t IntID>
  class string_to_int_assign_callable : public base_callable {
  public:
    string_to_int_assign_callable()
        : base_callable(
              ndt::callable_type::make(ndt::type(string_id), {ndt::type(IntID)}, {"error_mode"},
                                       {ndt::make_type<ndt::option_type>(ndt::make_type<assign_error_mode>())})) {}

    ndt::type resolve(base_callable *DYND_UNUSED(caller), char *DYND_UNUSED(data), call_graph &cg,
                      const ndt::type &dst_tp, size_t DYND_UNUSED(nsrc), const ndt::type *DYND_UNUSED(src_tp),
                      size_t DYND_UNUSED(nkwd), const array *DYND_UNUSED(kwds),
                      const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars)) {
      type_id_t dst_id = dst_tp.get_id();
      size_t string_size = 0;
      string_encoding_t string_encoding = string_encoding_ascii;
      if (dst_id == fixed_string_id) {
        string_size = dst_tp.extended<ndt::fixed_string_type>()->get_size();
        string_encoding = dst_tp.extended<ndt::fixed_string_type>()->get_encoding();
      }

      cg.push_back([dst_id, string_size, string_encoding](kernel_builder *ckb, kernel_request_t kernreq,
                                                          const char *dst_arrmeta, size_t DYND_UNUSED(nsrc),
                                                          const char *const *DYND_UNUSED(src_arrmeta)) {

        ndt::type string_tp = dst_id == fixed_string_id
                                  ? ndt::make_type<ndt::fixed_string_type>(string_size, string_encoding)
                                  : ndt::make_type<ndt::string_type>();
        ckb->emplace_back<
            detail::assignment_kernel<IntID, int_kind_id, string_id, string_kind_id, assign_error_default>>(
            kernreq, string_tp, dst_arrmeta);
      });

      return dst_tp;
    }
  };

  template <>
  class assign_callable<float64_id, string_id> : public base_callable {
  public:
    assign_callable()
        : base_callable(
              ndt::callable_type::make(ndt::type(float64_id), {ndt::type(string_id)}, {"error_mode"},
                                       {ndt::make_type<ndt::option_type>(ndt::make_type<assign_error_mode>())})) {}

    ndt::type resolve(base_callable *DYND_UNUSED(caller), char *DYND_UNUSED(data), call_graph &cg,
                      const ndt::type &dst_tp, size_t DYND_UNUSED(nsrc), const ndt::type *DYND_UNUSED(src_tp),
                      size_t DYND_UNUSED(nkwd), const array *kwds,
                      const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars)) {
      assign_error_mode error_mode =
          (kwds == NULL || kwds[0].is_na()) ? assign_error_default : kwds[0].as<assign_error_mode>();

      cg.push_back([error_mode](kernel_builder *ckb, kernel_request_t kernreq, const char *DYND_UNUSED(dst_arrmeta),
                                size_t DYND_UNUSED(nsrc), const char *const *src_arrmeta) {
        ckb->emplace_back<
            detail::assignment_kernel<float64_id, float_kind_id, string_id, string_kind_id, assign_error_nocheck>>(
            kernreq, ndt::type(string_id), src_arrmeta[0], error_mode);
      });

      return dst_tp;
    }
  };

  template <>
  class assign_callable<fixed_string_id, string_id> : public base_callable {
  public:
    assign_callable()
        : base_callable(
              ndt::callable_type::make(ndt::type(fixed_string_id), {ndt::type(string_id)}, {"error_mode"},
                                       {ndt::make_type<ndt::option_type>(ndt::make_type<assign_error_mode>())})) {}

    ndt::type resolve(base_callable *DYND_UNUSED(caller), char *DYND_UNUSED(data), call_graph &cg,
                      const ndt::type &dst_tp, size_t DYND_UNUSED(nsrc), const ndt::type *src_tp,
                      size_t DYND_UNUSED(nkwd), const array *kwds,
                      const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars)) {
      assign_error_mode error_mode = kwds[0].is_na() ? assign_error_default : kwds[0].as<assign_error_mode>();
      const ndt::base_string_type *src_fs = src_tp[0].extended<ndt::base_string_type>();

      size_t dst_data_size = dst_tp.get_data_size();
      string_encoding_t dst_encoding = dst_tp.extended<ndt::fixed_string_type>()->get_encoding();
      string_encoding_t src0_encoding = src_fs->get_encoding();
      cg.push_back([dst_data_size, dst_encoding, src0_encoding,
                    error_mode](kernel_builder *ckb, kernel_request_t kernreq, const char *DYND_UNUSED(dst_arrmeta),
                                size_t DYND_UNUSED(nsrc), const char *const *DYND_UNUSED(src_arrmeta)) {
        ckb->emplace_back<detail::assignment_kernel<fixed_string_id, string_kind_id, string_id, string_kind_id,
                                                    assign_error_nocheck>>(
            kernreq, get_next_unicode_codepoint_function(src0_encoding, error_mode),
            get_append_unicode_codepoint_function(dst_encoding, error_mode), dst_data_size,
            error_mode != assign_error_nocheck);
      });

      return dst_tp;
    }
  };

  template <>
  class assign_callable<fixed_string_id, fixed_string_id> : public base_callable {
  public:
    assign_callable()
        : base_callable(
              ndt::callable_type::make(ndt::type(fixed_string_id), {ndt::type(fixed_string_id)}, {"error_mode"},
                                       {ndt::make_type<ndt::option_type>(ndt::make_type<assign_error_mode>())})) {}

    ndt::type resolve(base_callable *DYND_UNUSED(caller), char *DYND_UNUSED(data), call_graph &cg,
                      const ndt::type &dst_tp, size_t DYND_UNUSED(nsrc), const ndt::type *src_tp,
                      size_t DYND_UNUSED(nkwd), const array *kwds,
                      const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars)) {
      assign_error_mode error_mode = kwds[0].is_na() ? assign_error_default : kwds[0].as<assign_error_mode>();

      cg.push_back([=](kernel_builder *ckb, kernel_request_t kernreq, const char *DYND_UNUSED(dst_arrmeta),
                       size_t DYND_UNUSED(nsrc), const char *const *DYND_UNUSED(src_arrmeta)) {
        const ndt::fixed_string_type *src_fs = src_tp[0].extended<ndt::fixed_string_type>();
        ckb->emplace_back<detail::assignment_kernel<fixed_string_id, string_kind_id, fixed_string_id, string_kind_id,
                                                    assign_error_nocheck>>(
            kernreq, get_next_unicode_codepoint_function(src_fs->get_encoding(), error_mode),
            get_append_unicode_codepoint_function(dst_tp.extended<ndt::fixed_string_type>()->get_encoding(),
                                                  error_mode),
            dst_tp.get_data_size(), src_fs->get_data_size(), error_mode != assign_error_nocheck);
      });

      return dst_tp;
    }
  };

  template <>
  class assign_callable<string_id, int_kind_id> : public base_callable {
  public:
    assign_callable()
        : base_callable(
              ndt::callable_type::make(ndt::type(string_id), {ndt::type(int_kind_id)}, {"error_mode"},
                                       {ndt::make_type<ndt::option_type>(ndt::make_type<assign_error_mode>())})) {}

    ndt::type resolve(base_callable *DYND_UNUSED(caller), char *DYND_UNUSED(data), call_graph &cg,
                      const ndt::type &dst_tp, size_t DYND_UNUSED(nsrc), const ndt::type *src_tp,
                      size_t DYND_UNUSED(nkwd), const array *DYND_UNUSED(kwds),
                      const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars)) {
      cg.push_back([=](kernel_builder *ckb, kernel_request_t kernreq, const char *dst_arrmeta, size_t DYND_UNUSED(nsrc),
                       const char *const *DYND_UNUSED(src_arrmeta)) {
        ckb->emplace_back<
            detail::assignment_kernel<string_id, string_kind_id, int8_id, int_kind_id, assign_error_nocheck>>(
            kernreq, dst_tp, src_tp[0].get_id(), dst_arrmeta);
      });

      return dst_tp;
    }
  };

  template <>
  class assign_callable<string_id, char_id> : public base_callable {
  public:
    assign_callable()
        : base_callable(
              ndt::callable_type::make(ndt::type(string_id), {ndt::type(char_id)}, {"error_mode"},
                                       {ndt::make_type<ndt::option_type>(ndt::make_type<assign_error_mode>())})) {}

    ndt::type resolve(base_callable *DYND_UNUSED(caller), char *DYND_UNUSED(data), call_graph &cg,
                      const ndt::type &dst_tp, size_t DYND_UNUSED(nsrc), const ndt::type *src_tp,
                      size_t DYND_UNUSED(nkwd), const array *kwds,
                      const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars)) {
      string_encoding_t dst_encoding = dst_tp.extended<ndt::base_string_type>()->get_encoding();
      string_encoding_t src0_encoding = src_tp[0].extended<ndt::char_type>()->get_encoding();
      size_t src0_data_size = src_tp[0].get_data_size();
      assign_error_mode error_mode = kwds[0].is_na() ? assign_error_default : kwds[0].as<assign_error_mode>();
      cg.push_back([dst_encoding, src0_data_size, src0_encoding,
                    error_mode](kernel_builder *ckb, kernel_request_t kernreq, const char *DYND_UNUSED(dst_arrmeta),
                                size_t DYND_UNUSED(nsrc), const char *const *DYND_UNUSED(src_arrmeta)) {
        ckb->emplace_back<detail::assignment_kernel<string_id, string_kind_id, fixed_string_id, string_kind_id,
                                                    assign_error_nocheck>>(
            kernreq, dst_encoding, src0_encoding, src0_data_size,
            get_next_unicode_codepoint_function(src0_encoding, error_mode),
            get_append_unicode_codepoint_function(dst_encoding, error_mode));
      });

      return dst_tp;
    }
  };

  template <>
  class assign_callable<type_id, string_id> : public base_callable {
  public:
    assign_callable()
        : base_callable(
              ndt::callable_type::make(ndt::type(type_id), {ndt::type(string_id)}, {"error_mode"},
                                       {ndt::make_type<ndt::option_type>(ndt::make_type<assign_error_mode>())})) {}

    ndt::type resolve(base_callable *DYND_UNUSED(caller), char *DYND_UNUSED(data), call_graph &cg,
                      const ndt::type &dst_tp, size_t DYND_UNUSED(nsrc), const ndt::type *DYND_UNUSED(src_tp),
                      size_t DYND_UNUSED(nkwd), const array *DYND_UNUSED(kwds),
                      const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars)) {
      cg.push_back([](kernel_builder *ckb, kernel_request_t kernreq, const char *DYND_UNUSED(dst_arrmeta),
                      size_t DYND_UNUSED(nsrc), const char *const *src_arrmeta) {
        ckb->emplace_back<
            detail::assignment_kernel<type_id, scalar_kind_id, string_id, string_kind_id, assign_error_nocheck>>(
            kernreq, ndt::type(string_id), src_arrmeta[0]);
      });

      return dst_tp;
    }
  };

  template <>
  class assign_callable<string_id, fixed_string_id> : public base_callable {
  public:
    assign_callable()
        : base_callable(
              ndt::callable_type::make(ndt::type(string_id), {ndt::type(fixed_string_id)}, {"error_mode"},
                                       {ndt::make_type<ndt::option_type>(ndt::make_type<assign_error_mode>())})) {}

    ndt::type resolve(base_callable *DYND_UNUSED(caller), char *DYND_UNUSED(data), call_graph &cg,
                      const ndt::type &dst_tp, size_t DYND_UNUSED(nsrc), const ndt::type *src_tp,
                      size_t DYND_UNUSED(nkwd), const array *kwds,
                      const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars)) {
      assign_error_mode error_mode = kwds[0].is_na() ? assign_error_default : kwds[0].as<assign_error_mode>();
      string_encoding_t dst_encoding = dst_tp.extended<ndt::base_string_type>()->get_encoding();
      size_t src0_data_size = src_tp[0].get_data_size();
      string_encoding_t src0_encoding = src_tp[0].extended<ndt::base_string_type>()->get_encoding();

      cg.push_back([error_mode, dst_encoding, src0_data_size,
                    src0_encoding](kernel_builder *ckb, kernel_request_t kernreq, const char *DYND_UNUSED(dst_arrmeta),
                                   size_t DYND_UNUSED(nsrc), const char *const *DYND_UNUSED(src_arrmeta)) {
        ckb->emplace_back<detail::assignment_kernel<string_id, string_kind_id, fixed_string_id, string_kind_id,
                                                    assign_error_nocheck>>(
            kernreq, dst_encoding, src0_encoding, src0_data_size,
            get_next_unicode_codepoint_function(src0_encoding, error_mode),
            get_append_unicode_codepoint_function(dst_encoding, error_mode));
      });

      return dst_tp;
    }
  };

  template <>
  class assign_callable<float32_id, string_id> : public base_callable {
  public:
    assign_callable()
        : base_callable(
              ndt::callable_type::make(ndt::type(float32_id), {ndt::type(string_id)}, {"error_mode"},
                                       {ndt::make_type<ndt::option_type>(ndt::make_type<assign_error_mode>())})) {}

    ndt::type resolve(base_callable *DYND_UNUSED(caller), char *DYND_UNUSED(data), call_graph &cg,
                      const ndt::type &dst_tp, size_t DYND_UNUSED(nsrc), const ndt::type *src_tp,
                      size_t DYND_UNUSED(nkwd), const array *kwds,
                      const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars)) {
      assign_error_mode error_mode = kwds[0].is_na() ? assign_error_default : kwds[0].as<assign_error_mode>();

      cg.push_back([=](kernel_builder *ckb, kernel_request_t kernreq, const char *DYND_UNUSED(dst_arrmeta),
                       size_t DYND_UNUSED(nsrc), const char *const *src_arrmeta) {
        ckb->emplace_back<
            detail::assignment_kernel<float32_id, float_kind_id, string_id, string_kind_id, assign_error_nocheck>>(
            kernreq, src_tp[0], src_arrmeta[0], error_mode);
      });

      return dst_tp;
    }
  };

  template <>
  class assign_callable<string_id, type_id> : public base_callable {
  public:
    assign_callable()
        : base_callable(
              ndt::callable_type::make(ndt::type(string_id), {ndt::type(type_id)}, {"error_mode"},
                                       {ndt::make_type<ndt::option_type>(ndt::make_type<assign_error_mode>())})) {}

    ndt::type resolve(base_callable *DYND_UNUSED(caller), char *DYND_UNUSED(data), call_graph &cg,
                      const ndt::type &dst_tp, size_t DYND_UNUSED(nsrc), const ndt::type *DYND_UNUSED(src_tp),
                      size_t DYND_UNUSED(nkwd), const array *DYND_UNUSED(kwds),
                      const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars)) {
      cg.push_back([](kernel_builder *ckb, kernel_request_t kernreq, const char *dst_arrmeta, size_t DYND_UNUSED(nsrc),
                      const char *const *DYND_UNUSED(src_arrmeta)) {
        ckb->emplace_back<
            detail::assignment_kernel<string_id, string_kind_id, type_id, scalar_kind_id, assign_error_nocheck>>(
            kernreq, ndt::type(string_id), dst_arrmeta);
      });

      return dst_tp;
    }
  };

  template <>
  class assign_callable<char_id, string_id> : public base_callable {
  public:
    assign_callable()
        : base_callable(
              ndt::callable_type::make(ndt::type(char_id), {ndt::type(string_id)}, {"error_mode"},
                                       {ndt::make_type<ndt::option_type>(ndt::make_type<assign_error_mode>())})) {}

    ndt::type resolve(base_callable *DYND_UNUSED(caller), char *DYND_UNUSED(data), call_graph &cg,
                      const ndt::type &dst_tp, size_t DYND_UNUSED(nsrc), const ndt::type *src_tp,
                      size_t DYND_UNUSED(nkwd), const array *kwds,
                      const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars)) {
      assign_error_mode error_mode = kwds[0].is_na() ? assign_error_default : kwds[0].as<assign_error_mode>();
      const ndt::base_string_type *src_fs = src_tp[0].extended<ndt::base_string_type>();
      size_t dst_data_size = dst_tp.get_data_size();
      string_encoding_t dst_encoding = dst_tp.extended<ndt::char_type>()->get_encoding();
      string_encoding_t src0_encoding = src_fs->get_encoding();
      cg.push_back([error_mode, dst_data_size, dst_encoding,
                    src0_encoding](kernel_builder *ckb, kernel_request_t kernreq, const char *DYND_UNUSED(dst_arrmeta),
                                   size_t DYND_UNUSED(nsrc), const char *const *DYND_UNUSED(src_arrmeta)) {
        ckb->emplace_back<detail::assignment_kernel<fixed_string_id, string_kind_id, string_id, string_kind_id,
                                                    assign_error_nocheck>>(
            kernreq, get_next_unicode_codepoint_function(src0_encoding, error_mode),
            get_append_unicode_codepoint_function(dst_encoding, error_mode), dst_data_size,
            error_mode != assign_error_nocheck);
      });

      return dst_tp;
    }
  };

  template <>
  class assign_callable<pointer_id, pointer_id> : public base_callable {
  public:
    assign_callable()
        : base_callable(
              ndt::callable_type::make(ndt::type(pointer_id), {ndt::type(pointer_id)}, {"error_mode"},
                                       {ndt::make_type<ndt::option_type>(ndt::make_type<assign_error_mode>())})) {}

    ndt::type resolve(base_callable *DYND_UNUSED(caller), char *DYND_UNUSED(data), call_graph &cg,
                      const ndt::type &dst_tp, size_t DYND_UNUSED(nsrc), const ndt::type *DYND_UNUSED(src_tp),
                      size_t DYND_UNUSED(nkwd), const array *DYND_UNUSED(kwds),
                      const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars)) {
      cg.push_back([](kernel_builder *ckb, kernel_request_t kernreq, const char *dst_arrmeta, size_t DYND_UNUSED(nsrc),
                      const char *const *src_arrmeta) {
        ckb->emplace_back<assignment_kernel<pointer_id, pointer_id>>(kernreq);

        const char *child_src_arrmeta = src_arrmeta[0] + sizeof(pointer_type_arrmeta);
        ckb->instantiate(kernel_request_single, dst_arrmeta, 1, &child_src_arrmeta);
      });

      return dst_tp;
    }
  };

  template <>
  class assign_callable<option_id, option_id> : public base_callable {
  public:
    assign_callable()
        : base_callable(
              ndt::callable_type::make(ndt::type(option_id), {ndt::type(option_id)}, {"error_mode"},
                                       {ndt::make_type<ndt::option_type>(ndt::make_type<assign_error_mode>())})) {}

    ndt::type resolve(base_callable *DYND_UNUSED(caller), char *DYND_UNUSED(data), call_graph &cg,
                      const ndt::type &dst_tp, size_t DYND_UNUSED(nsrc), const ndt::type *src_tp, size_t nkwd,
                      const array *kwds, const std::map<std::string, ndt::type> &tp_vars) {
      cg.push_back([](kernel_builder *ckb, kernel_request_t kernreq, const char *dst_arrmeta, size_t nsrc,
                      const char *const *src_arrmeta) {
        intptr_t ckb_offset = ckb->size();
        intptr_t root_ckb_offset = ckb_offset;
        typedef detail::assignment_kernel<option_id, any_kind_id, option_id, any_kind_id, assign_error_nocheck>
            self_type;

        ckb->emplace_back<self_type>(kernreq);
        ckb_offset = ckb->size();
        // instantiate src_is_avail
        ckb->instantiate(kernreq | kernel_request_data_only, nullptr, nsrc, src_arrmeta);

        ckb_offset = ckb->size();
        // instantiate dst_assign_na
        ckb->reserve(ckb_offset + sizeof(kernel_prefix));
        self_type *self = ckb->get_at<self_type>(root_ckb_offset);
        self->m_dst_assign_na_offset = ckb_offset - root_ckb_offset;
        ckb->instantiate(kernreq | kernel_request_data_only, dst_arrmeta, nsrc, nullptr);

        ckb_offset = ckb->size();
        // instantiate value_assign
        ckb->reserve(ckb_offset + sizeof(kernel_prefix));
        self = ckb->get_at<self_type>(root_ckb_offset);
        self->m_value_assign_offset = ckb_offset - root_ckb_offset;
        ckb->instantiate(kernreq | kernel_request_data_only, dst_arrmeta, 1, src_arrmeta);
      });

      is_na->resolve(this, nullptr, cg, ndt::make_type<bool1>(), 1, src_tp, nkwd, kwds, tp_vars);
      assign_na->resolve(this, nullptr, cg, dst_tp, 1, nullptr, nkwd, kwds, tp_vars);

      const ndt::type &src_val_tp = src_tp[0].extended<ndt::option_type>()->get_value_type();
      assign->resolve(this, nullptr, cg, dst_tp.extended<ndt::option_type>()->get_value_type(), 1, &src_val_tp, nkwd,
                      kwds, tp_vars);

      return dst_tp;
    }
  };

  template <>
  class assign_callable<option_id, float_kind_id> : public base_callable {
  public:
    assign_callable()
        : base_callable(
              ndt::callable_type::make(ndt::type(option_id), {ndt::type(float_kind_id)}, {"error_mode"},
                                       {ndt::make_type<ndt::option_type>(ndt::make_type<assign_error_mode>())})) {}

    ndt::type resolve(base_callable *DYND_UNUSED(caller), char *DYND_UNUSED(data), call_graph &cg,
                      const ndt::type &dst_tp, size_t nsrc, const ndt::type *src_tp, size_t nkwd, const array *kwds,
                      const std::map<std::string, ndt::type> &tp_vars) {
      ndt::type src_tp_as_option = ndt::make_type<ndt::option_type>(src_tp[0]);
      static callable f = make_callable<assign_callable<option_id, option_id>>();

      f->resolve(this, nullptr, cg, dst_tp, nsrc, &src_tp_as_option, nkwd, kwds, tp_vars);

      return dst_tp;
    }

    void instantiate(call_node *&node, char *DYND_UNUSED(data), kernel_builder *ckb, const ndt::type &dst_tp,
                     const char *dst_arrmeta, intptr_t nsrc, const ndt::type *src_tp, const char *const *src_arrmeta,
                     kernel_request_t kernreq, intptr_t nkwd, const nd::array *kwds,
                     const std::map<std::string, ndt::type> &tp_vars) {
      // Deal with some float32 to option[T] conversions where any NaN is
      // interpreted
      // as NA.
      ndt::type src_tp_as_option = ndt::make_type<ndt::option_type>(src_tp[0]);
      callable f = make_callable<assign_callable<option_id, option_id>>();
      f->instantiate(node, NULL, ckb, dst_tp, dst_arrmeta, nsrc, &src_tp_as_option, src_arrmeta, kernreq, nkwd, kwds,
                     tp_vars);
    }
  };

  template <>
  class assign_callable<option_id, string_id> : public base_callable {
  public:
    assign_callable()
        : base_callable(
              ndt::callable_type::make(ndt::type(option_id), {ndt::type(string_id)}, {"error_mode"},
                                       {ndt::make_type<ndt::option_type>(ndt::make_type<assign_error_mode>())})) {}

    ndt::type resolve(base_callable *DYND_UNUSED(caller), char *DYND_UNUSED(data), call_graph &cg,
                      const ndt::type &dst_tp, size_t nsrc, const ndt::type *src_tp, size_t nkwd, const array *kwds,
                      const std::map<std::string, ndt::type> &tp_vars) {
      assign_error_mode error_mode = kwds[0].is_na() ? assign_error_default : kwds[0].as<assign_error_mode>();

      type_id_t tid = dst_tp.get_dtype().extended<ndt::option_type>()->get_value_type().get_id();
      switch (tid) {
      case bool_id:
        cg.push_back([](kernel_builder *ckb, kernel_request_t kernreq, const char *DYND_UNUSED(dst_arrmeta),
                        size_t DYND_UNUSED(nsrc), const char *const *DYND_UNUSED(src_arrmeta)) {
          ckb->emplace_back<detail::string_to_option_bool_ck>(kernreq);
        });
        break;
      case int8_id:
      case int16_id:
      case int32_id:
      case int64_id:
      case int128_id:
      case float16_id:
      case float32_id:
      case float64_id:
        cg.push_back([tid, error_mode](kernel_builder *ckb, kernel_request_t kernreq,
                                       const char *DYND_UNUSED(dst_arrmeta), size_t DYND_UNUSED(nsrc),
                                       const char *const *DYND_UNUSED(src_arrmeta)) {
          ckb->emplace_back<detail::string_to_option_number_ck>(kernreq, tid, error_mode);
        });
        break;
      case string_id:
        cg.push_back([](kernel_builder *ckb, kernel_request_t kernreq, const char *dst_arrmeta, size_t nsrc,
                        const char *const *src_arrmeta) { ckb->instantiate(kernreq, dst_arrmeta, nsrc, src_arrmeta); });
        break;
      default:
        cg.push_back([](kernel_builder *ckb, kernel_request_t kernreq, const char *dst_arrmeta, size_t nsrc,
                        const char *const *src_arrmeta) {
          // Fall back to an adaptor that checks for a few standard
          // missing value tokens, then uses the standard value assignment
          intptr_t ckb_offset = ckb->size();
          intptr_t root_ckb_offset = ckb_offset;
          ckb->emplace_back<detail::string_to_option_tp_ck>(kernreq);

          ckb_offset = ckb->size();
          // First child ckernel is the value assignment
          ckb->instantiate(kernreq | kernel_request_data_only, dst_arrmeta, nsrc, src_arrmeta);
          ckb_offset = ckb->size();
          // Re-acquire self because the address may have changed
          detail::string_to_option_tp_ck *self = ckb->get_at<detail::string_to_option_tp_ck>(root_ckb_offset);
          // Second child ckernel is the NA assignment
          self->m_dst_assign_na_offset = ckb_offset - root_ckb_offset;
          ckb->instantiate(kernreq | kernel_request_data_only, dst_arrmeta, nsrc, src_arrmeta);
          ckb_offset = ckb->size();
        });
        break;
      }

      assign->resolve(this, nullptr, cg, dst_tp.extended<ndt::option_type>()->get_value_type(), nsrc, src_tp, nkwd,
                      kwds, tp_vars);
      assign_na->resolve(this, nullptr, cg, dst_tp, nsrc, src_tp, nkwd, kwds, tp_vars);

      return dst_tp;
    }
  };

  template <>
  class assign_callable<tuple_id, tuple_id> : public base_callable {
  public:
    assign_callable()
        : base_callable(
              ndt::callable_type::make(ndt::type(tuple_id), {ndt::type(tuple_id)}, {"error_mode"},
                                       {ndt::make_type<ndt::option_type>(ndt::make_type<assign_error_mode>())})) {}

    ndt::type resolve(base_callable *DYND_UNUSED(caller), char *DYND_UNUSED(data), call_graph &cg,
                      const ndt::type &dst_tp, size_t DYND_UNUSED(nsrc), const ndt::type *src_tp, size_t nkwd,
                      const array *kwds, const std::map<std::string, ndt::type> &tp_vars) {
      auto dst_sd = dst_tp.extended<ndt::tuple_type>();
      auto src_sd = src_tp[0].extended<ndt::tuple_type>();

      intptr_t field_count = dst_sd->get_field_count();
      if (field_count != src_sd->get_field_count()) {
        std::stringstream ss;
        ss << "cannot assign dynd " << src_tp[0] << " to " << dst_tp
           << " because they have different numbers of fields";
        throw type_error(ss.str());
      }

      std::array<uintptr_t, 8> dst_arrmeta_offsets;
      std::array<uintptr_t, 8> src_arrmeta_offsets;
      for (int i = 0; i < field_count; ++i) {
        src_arrmeta_offsets[i] = src_sd->get_arrmeta_offsets()[i];
        dst_arrmeta_offsets[i] = dst_sd->get_arrmeta_offsets()[i];
      }

      cg.push_back([field_count, dst_arrmeta_offsets,
                    src_arrmeta_offsets](kernel_builder *ckb, kernel_request_t kernreq, const char *dst_arrmeta,
                                         size_t DYND_UNUSED(nsrc), const char *const *src_arrmeta) {
        shortvector<const char *> src_fields_arrmeta(field_count);
        for (intptr_t i = 0; i != field_count; ++i) {
          src_fields_arrmeta[i] = src_arrmeta[0] + src_arrmeta_offsets[i];
        }

        shortvector<const char *> dst_fields_arrmeta(field_count);
        for (intptr_t i = 0; i != field_count; ++i) {
          dst_fields_arrmeta[i] = dst_arrmeta + dst_arrmeta_offsets[i];
        }

        const uintptr_t *dst_data_offsets = reinterpret_cast<const uintptr_t *>(dst_arrmeta);
        const uintptr_t *src_data_offsets = reinterpret_cast<const uintptr_t *>(src_arrmeta[0]);

        intptr_t self_offset = ckb->size();
        ckb->emplace_back<nd::tuple_unary_op_ck>(kernreq);
        nd::tuple_unary_op_ck *self = ckb->get_at<nd::tuple_unary_op_ck>(self_offset);
        self->m_fields.resize(field_count);
        for (intptr_t i = 0; i < field_count; ++i) {
          self = ckb->get_at<nd::tuple_unary_op_ck>(self_offset);
          nd::tuple_unary_op_item &field = self->m_fields[i];
          field.child_kernel_offset = ckb->size() - self_offset;
          field.dst_data_offset = dst_data_offsets[i];
          field.src_data_offset = src_data_offsets[i];
          ckb->instantiate(kernel_request_single, dst_fields_arrmeta[i], 1, &src_fields_arrmeta[i]);
        }
      });

      const std::vector<ndt::type> &dst_field_tp = dst_sd->get_field_types();
      const std::vector<ndt::type> &src_field_tp = src_sd->get_field_types();
      for (intptr_t i = 0; i < field_count; ++i) {
        assign->resolve(this, nullptr, cg, dst_field_tp[i], 1, &src_field_tp[i], nkwd, kwds, tp_vars);
      }

      return dst_tp;
    }
  };

  template <>
  class assign_callable<struct_id, struct_id> : public base_callable {
  public:
    assign_callable()
        : base_callable(
              ndt::callable_type::make(ndt::type(struct_id), {ndt::type(struct_id)}, {"error_mode"},
                                       {ndt::make_type<ndt::option_type>(ndt::make_type<assign_error_mode>())})) {}

    ndt::type resolve(base_callable *DYND_UNUSED(caller), char *DYND_UNUSED(data), call_graph &cg,
                      const ndt::type &dst_tp, size_t DYND_UNUSED(nsrc), const ndt::type *src_tp, size_t nkwd,
                      const array *kwds, const std::map<std::string, ndt::type> &tp_vars) {
      const ndt::struct_type *dst_sd = dst_tp.extended<ndt::struct_type>();
      const ndt::struct_type *src_sd = src_tp[0].extended<ndt::struct_type>();
      intptr_t field_count = dst_sd->get_field_count();
      std::array<intptr_t, 8> src_permutation;
      std::array<uintptr_t, 8> src_fields_arrmeta_offsets;
      std::array<uintptr_t, 8> dst_arrmeta_offsets;

      if (field_count != src_sd->get_field_count()) {
        std::stringstream ss;
        ss << "cannot assign dynd struct " << src_tp[0] << " to " << dst_tp;
        ss << " because they have different numbers of fields";
        throw std::runtime_error(ss.str());
      }

      const std::vector<ndt::type> &src_fields_tp_orig = src_sd->get_field_types();
      const std::vector<uintptr_t> &src_arrmeta_offsets_orig = src_sd->get_arrmeta_offsets();
      std::vector<ndt::type> src_fields_tp(field_count);

      // Match up the fields
      for (intptr_t i = 0; i != field_count; ++i) {
        const std::string &dst_name = dst_sd->get_field_name(i);
        intptr_t src_i = src_sd->get_field_index(dst_name);
        if (src_i < 0) {
          std::stringstream ss;
          ss << "cannot assign dynd struct " << src_tp[0] << " to " << dst_tp;
          ss << " because they have different field names";
          throw std::runtime_error(ss.str());
        }
        src_fields_tp[i] = src_fields_tp_orig[src_i];
        //        src_data_offsets[i] = src_data_offsets_orig[src_i];
        src_fields_arrmeta_offsets[i] = src_arrmeta_offsets_orig[src_i];
        src_permutation[i] = src_i;
      }

      const std::vector<ndt::type> &dst_fields_tp = dst_sd->get_field_types();
      const std::vector<uintptr_t> &dst_arrmeta_offsets_vec = dst_sd->get_arrmeta_offsets();
      for (intptr_t i = 0; i < field_count; ++i) {
        dst_arrmeta_offsets[i] = dst_arrmeta_offsets_vec[i];
      }

      cg.push_back([field_count, src_permutation, src_fields_arrmeta_offsets,
                    dst_arrmeta_offsets](kernel_builder *ckb, kernel_request_t kernreq, const char *dst_arrmeta,
                                         size_t DYND_UNUSED(nsrc), const char *const *src_arrmeta) {
        const uintptr_t *src_data_offsets_orig = reinterpret_cast<const uintptr_t *>(src_arrmeta[0]);
        shortvector<uintptr_t> src_data_offsets(field_count);
        shortvector<const char *> src_fields_arrmeta(field_count);

        // Match up the fields
        for (intptr_t i = 0; i != field_count; ++i) {
          intptr_t src_i = src_permutation[i];
          src_data_offsets[i] = src_data_offsets_orig[src_i];
          src_fields_arrmeta[i] = src_arrmeta[0] + src_fields_arrmeta_offsets[src_i];
        }

        shortvector<const char *> dst_fields_arrmeta(field_count);
        for (intptr_t i = 0; i != field_count; ++i) {
          dst_fields_arrmeta[i] = dst_arrmeta + dst_arrmeta_offsets[i];
        }

        const uintptr_t *dst_offsets = reinterpret_cast<const uintptr_t *>(dst_arrmeta);

        intptr_t self_offset = ckb->size();
        ckb->emplace_back<nd::tuple_unary_op_ck>(kernreq);
        nd::tuple_unary_op_ck *self = ckb->get_at<nd::tuple_unary_op_ck>(self_offset);
        self->m_fields.resize(field_count);
        for (intptr_t i = 0; i < field_count; ++i) {
          self = ckb->get_at<nd::tuple_unary_op_ck>(self_offset);
          nd::tuple_unary_op_item &field = self->m_fields[i];
          field.child_kernel_offset = ckb->size() - self_offset;
          field.dst_data_offset = dst_offsets[i];
          field.src_data_offset = src_data_offsets[i];
          ckb->instantiate(kernel_request_single, dst_fields_arrmeta[i], 1, &src_fields_arrmeta[i]);
        }
      });

      for (intptr_t i = 0; i < field_count; ++i) {
        nd::assign->resolve(this, nullptr, cg, dst_fields_tp[i], 1, &src_fields_tp[i], nkwd, kwds, tp_vars);
      }

      return dst_tp;
    }
  };

  class option_to_value_callable : public base_callable {
  public:
    option_to_value_callable() : base_callable(ndt::type("(Any) -> Any")) {}

    ndt::type resolve(base_callable *DYND_UNUSED(caller), char *DYND_UNUSED(data), call_graph &cg,
                      const ndt::type &dst_tp, size_t DYND_UNUSED(nsrc), const ndt::type *src_tp, size_t nkwd,
                      const array *kwds, const std::map<std::string, ndt::type> &tp_vars) {
      cg.push_back([](kernel_builder *ckb, kernel_request_t kernreq, const char *dst_arrmeta, size_t nsrc,
                      const char *const *src_arrmeta) {
        intptr_t ckb_offset = ckb->size();
        intptr_t root_ckb_offset = ckb_offset;
        ckb->emplace_back<option_to_value_ck>(kernreq);

        ckb->instantiate(kernreq | kernel_request_data_only, nullptr, nsrc, src_arrmeta);

        ckb_offset = ckb->size();
        // instantiate value_assign
        ckb->reserve(ckb_offset + sizeof(kernel_prefix));
        option_to_value_ck *self = ckb->get_at<option_to_value_ck>(root_ckb_offset);
        self->m_value_assign_offset = ckb_offset - root_ckb_offset;

        ckb->instantiate(kernreq | kernel_request_data_only, dst_arrmeta, 1, src_arrmeta);
      });

      is_na->resolve(this, nullptr, cg, ndt::make_type<bool1>(), 1, src_tp, nkwd, kwds, tp_vars);
      const ndt::type &src_val_tp = src_tp[0].extended<ndt::option_type>()->get_value_type();
      assign->resolve(this, nullptr, cg, dst_tp, 1, &src_val_tp, nkwd, kwds, tp_vars);

      return dst_tp;
    }
  };

  class adapt_assign_from_callable : public base_callable {
  public:
    adapt_assign_from_callable() : base_callable(ndt::type("(Any) -> Any")) {}

    ndt::type resolve(base_callable *DYND_UNUSED(caller), char *DYND_UNUSED(data), call_graph &DYND_UNUSED(cg),
                      const ndt::type &dst_tp, size_t DYND_UNUSED(nsrc), const ndt::type *DYND_UNUSED(src_tp),
                      size_t DYND_UNUSED(nkwd), const array *DYND_UNUSED(kwds),
                      const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars)) {
      return dst_tp;
    }

    /*
        void instantiate(call_node *&node, char *data, kernel_builder *ckb, const ndt::type &dst_tp,
                         const char *dst_arrmeta, intptr_t nsrc, const ndt::type *src_tp, const char *const
       *src_arrmeta,
                         kernel_request_t kernreq, intptr_t nkwd, const nd::array *kwds,
                         const std::map<std::string, ndt::type> &tp_vars) {
          intptr_t ckb_offset = ckb->size();
          const ndt::type &storage_tp = src_tp[0].storage_type();
          if (storage_tp.is_expression()) {
            const callable &forward = src_tp[0].extended<ndt::adapt_type>()->get_forward();

            intptr_t self_offset = ckb_offset;
            ckb->emplace_back<detail::adapt_assign_from_kernel>(kernreq, storage_tp.get_canonical_type());
            ckb_offset = ckb->size();

            nd::assign->instantiate(node, data, ckb, storage_tp.get_canonical_type(), dst_arrmeta, nsrc, &storage_tp,
                                    src_arrmeta, kernel_request_single, nkwd, kwds, tp_vars);
            ckb_offset = ckb->size();
            intptr_t forward_offset = ckb_offset - self_offset;
            ndt::type src_tp2[1] = {storage_tp.get_canonical_type()};
            forward->instantiate(node, data, ckb, dst_tp, dst_arrmeta, nsrc, src_tp2, src_arrmeta,
       kernel_request_single,
                                 nkwd, kwds, tp_vars);
            ckb_offset = ckb->size();
            ckb->get_at<detail::adapt_assign_from_kernel>(self_offset)->forward_offset = forward_offset;
          } else {
            const callable &forward = src_tp[0].extended<ndt::adapt_type>()->get_forward();

            ndt::type src_tp2[1] = {storage_tp.get_canonical_type()};
            forward->instantiate(node, data, ckb, dst_tp, dst_arrmeta, nsrc, src_tp2, src_arrmeta, kernreq, nkwd, kwds,
                                 tp_vars);
            ckb_offset = ckb->size();
          }
        }
    */
  };

  class adapt_assign_to_callable : public base_callable {
  public:
    adapt_assign_to_callable() : base_callable(ndt::type("(Any) -> Any")) {}

    ndt::type resolve(base_callable *DYND_UNUSED(caller), char *DYND_UNUSED(data), call_graph &cg,
                      const ndt::type &dst_tp, size_t nsrc, const ndt::type *DYND_UNUSED(src_tp), size_t nkwd,
                      const array *kwds, const std::map<std::string, ndt::type> &tp_vars) {
      const callable &inverse = dst_tp.extended<ndt::adapt_type>()->get_inverse();
      const ndt::type &value_tp = dst_tp.value_type();

      inverse->resolve(this, nullptr, cg, dst_tp.storage_type(), nsrc, &value_tp, nkwd, kwds, tp_vars);

      return dst_tp;
    }
  };

  class assignment_option_callable : public base_callable {
  public:
    assignment_option_callable() : base_callable(ndt::type("(Any) -> ?Any")) {}

    ndt::type resolve(base_callable *DYND_UNUSED(caller), char *DYND_UNUSED(data), call_graph &cg,
                      const ndt::type &dst_tp, size_t DYND_UNUSED(nsrc), const ndt::type *src_tp, size_t nkwd,
                      const array *kwds, const std::map<std::string, ndt::type> &tp_vars) {

      ndt::type val_dst_tp =
          dst_tp.get_id() == option_id ? dst_tp.extended<ndt::option_type>()->get_value_type() : dst_tp;
      ndt::type val_src_tp =
          src_tp[0].get_id() == option_id ? src_tp[0].extended<ndt::option_type>()->get_value_type() : src_tp[0];
      assign->resolve(this, nullptr, cg, val_dst_tp, 1, &val_src_tp, nkwd, kwds, tp_vars);

      return dst_tp;
    }

    void instantiate(call_node *&node, char *DYND_UNUSED(data), kernel_builder *ckb, const ndt::type &dst_tp,
                     const char *dst_arrmeta, intptr_t DYND_UNUSED(nsrc), const ndt::type *src_tp,
                     const char *const *src_arrmeta, kernel_request_t kernreq, intptr_t nkwd, const nd::array *kwds,
                     const std::map<std::string, ndt::type> &tp_vars) {
      ndt::type val_dst_tp =
          dst_tp.get_id() == option_id ? dst_tp.extended<ndt::option_type>()->get_value_type() : dst_tp;
      ndt::type val_src_tp =
          src_tp[0].get_id() == option_id ? src_tp[0].extended<ndt::option_type>()->get_value_type() : src_tp[0];
      assign->instantiate(node, NULL, ckb, val_dst_tp, dst_arrmeta, 1, &val_src_tp, src_arrmeta, kernreq, nkwd, kwds,
                          tp_vars);
    }
  };

} // namespace dynd::nd
} // namespace dynd
