//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/option.hpp>
#include <dynd/kernels/base_kernel.hpp>

namespace dynd {
namespace nd {
  namespace json {

    template <type_id_t RetTypeID>
    struct parse_kernel : base_kernel<parse_kernel<RetTypeID>> {
      typedef typename type_of<RetTypeID>::type ret_type;

      void single(char *ret, char *const *args)
      {
        const char *&begin = *reinterpret_cast<const char **>(args[0]);
        const char *&end = *reinterpret_cast<const char **>(args[1]);

        const char *nbegin, *nend;
        if (parse_number(begin, end, nbegin, nend)) {
          *reinterpret_cast<ret_type *>(ret) = dynd::parse<ret_type>(nbegin, nend);
        }
      }
    };

    template <>
    struct parse_kernel<option_type_id> : base_kernel<parse_kernel<option_type_id>> {
      intptr_t parse_offset;

      ~parse_kernel()
      {
        get_child()->destroy();
        get_child(parse_offset)->destroy();
      }

      void single(char *ret, char *const *args)
      {
        //        const char *saved_args[2] = {*reinterpret_cast<const char **>(args[0]),
        //                                   *reinterpret_cast<const char **>(args[1])};
        if (parse_na(*reinterpret_cast<const char **>(args[0]), *reinterpret_cast<const char **>(args[1]))) {
          get_child()->single(ret, nullptr);
        }
        else {
          get_child(parse_offset)->single(ret, args);
        }
      }

      static intptr_t instantiate(char *DYND_UNUSED(static_data), char *data, void *ckb, intptr_t ckb_offset,
                                  const ndt::type &dst_tp, const char *dst_arrmeta, intptr_t nsrc,
                                  const ndt::type *src_tp, const char *const *src_arrmeta, kernel_request_t kernreq,
                                  const eval::eval_context *DYND_UNUSED(ectx), intptr_t nkwd, const nd::array *kwds,
                                  const std::map<std::string, ndt::type> &tp_vars)
      {

        intptr_t self_offset = ckb_offset;
        make(ckb, kernreq, ckb_offset);

        ckb_offset =
            assign_na::get()->instantiate(assign_na::get()->static_data(), data, ckb, ckb_offset, dst_tp, dst_arrmeta,
                                          0, nullptr, nullptr, kernreq, NULL, nkwd, kwds, tp_vars);

        get_self(reinterpret_cast<ckernel_builder<kernel_request_host> *>(ckb), self_offset)->parse_offset =
            ckb_offset - self_offset;
        ckb_offset = parse::get()->instantiate(parse::get()->static_data(), data, ckb, ckb_offset,
                                               dst_tp.extended<ndt::option_type>()->get_value_type(), dst_arrmeta, nsrc,
                                               src_tp, src_arrmeta, kernreq, NULL, nkwd, kwds, tp_vars);

        return ckb_offset;
      }
    };

    template <>
    struct parse_kernel<fixed_dim_type_id> : base_kernel<parse_kernel<fixed_dim_type_id>> {
      ndt::type ret_tp;
      size_t size;
      intptr_t stride;

      ~parse_kernel() { get_child()->destroy(); }

      parse_kernel(const ndt::type &ret_tp, size_t size, intptr_t stride) : ret_tp(ret_tp), size(size), stride(stride)
      {
      }

      void single(char *ret, char *const *args)
      {
        if (!parse_token(args, "[")) {
          throw json_parse_error(args, "expected list starting with '['", ret_tp);
        }
        skip_whitespace(args);

        ckernel_prefix *child = get_child();
        for (size_t i = 0; i < size; ++i) {
          child->single(ret, args);
          if (i < size - 1 && !parse_token(args, ",")) {
            throw json_parse_error(args, "array is too short, expected ',' list item separator", ret_tp);
          }
          skip_whitespace(args);

          ret += stride;
        }

        if (!parse_token(args, "]")) {
          throw json_parse_error(args, "array is too long, expected list terminator ']'", ret_tp);
        }
      }

      static intptr_t instantiate(char *DYND_UNUSED(static_data), char *data, void *ckb, intptr_t ckb_offset,
                                  const ndt::type &dst_tp, const char *dst_arrmeta, intptr_t nsrc,
                                  const ndt::type *src_tp, const char *const *src_arrmeta, kernel_request_t kernreq,
                                  const eval::eval_context *DYND_UNUSED(ectx), intptr_t nkwd, const nd::array *kwds,
                                  const std::map<std::string, ndt::type> &tp_vars)
      {
        make(ckb, kernreq, ckb_offset, dst_tp, reinterpret_cast<const size_stride_t *>(dst_arrmeta)->dim_size,
             reinterpret_cast<const size_stride_t *>(dst_arrmeta)->stride);

        const ndt::type &child_dst_tp = dst_tp.extended<ndt::fixed_dim_type>()->get_element_type();
        return json::parse::get()->instantiate(json::parse::get()->static_data(), data, ckb, ckb_offset, child_dst_tp,
                                               dst_arrmeta + sizeof(ndt::fixed_dim_type::metadata_type), nsrc, src_tp,
                                               src_arrmeta, kernreq, NULL, nkwd, kwds, tp_vars);
      }
    };

  } // namespace dynd::nd::json
} // namespace dynd::nd

namespace ndt {

  template <type_id_t DstTypeID>
  struct traits<nd::json::parse_kernel<DstTypeID>> {
    static type equivalent() { return callable_type::make(DstTypeID, {make_type<char *>(), make_type<char **>()}); }
  };

} // namespace dynd::ndt
} // namespace dynd
