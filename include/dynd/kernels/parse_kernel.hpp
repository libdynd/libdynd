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

    inline bool _parse_number(const char *&rbegin, const char *&end)
    {
      const char *begin = rbegin;
      if (begin == end) {
        return false;
      }
      // Optional minus sign
      if (*begin == '-') {
        ++begin;
      }
      if (begin == end) {
        return false;
      }
      // Either '0' or a non-zero digit followed by digits
      if (*begin == '0') {
        ++begin;
      }
      else if ('1' <= *begin && *begin <= '9') {
        ++begin;
        while (begin < end && ('0' <= *begin && *begin <= '9')) {
          ++begin;
        }
      }
      else {
        return false;
      }
      // Optional decimal point, followed by one or more digits
      if (begin < end && *begin == '.') {
        if (++begin == end) {
          return false;
        }
        if (!('0' <= *begin && *begin <= '9')) {
          return false;
        }
        ++begin;
        while (begin < end && ('0' <= *begin && *begin <= '9')) {
          ++begin;
        }
      }
      // Optional exponent, followed by +/- and some digits
      if (begin < end && (*begin == 'e' || *begin == 'E')) {
        if (++begin == end) {
          return false;
        }
        // +/- is optional
        if (*begin == '+' || *begin == '-') {
          if (++begin == end) {
            return false;
          }
        }
        // At least one digit is required
        if (!('0' <= *begin && *begin <= '9')) {
          return false;
        }
        ++begin;
        while (begin < end && ('0' <= *begin && *begin <= '9')) {
          ++begin;
        }
      }

      end = begin;
      return true;
    }

    inline bool _parse_na(const char *&begin, const char *end)
    {
      size_t size = end - begin;

      if (size >= 4) {
        if (((begin[0] == 'N' || begin[0] == 'n') && (begin[1] == 'U' || begin[1] == 'u') &&
             (begin[2] == 'L' || begin[2] == 'l') && (begin[3] == 'L' || begin[3] == 'l'))) {
          begin += 4;
          return true;
        }
        if (begin[0] == 'N' && begin[1] == 'o' && begin[2] == 'n' && begin[3] == 'e') {
          begin += 4;
          return true;
        }
      }

      if (size >= 2 && begin[0] == 'N' && begin[1] == 'A') {
        begin += 2;
        return true;
      }

      if (size == 0) {
        return true;
      }

      return false;
    }

    template <type_id_t RetTypeID>
    struct parse_kernel : base_kernel<parse_kernel<RetTypeID>, 2> {
      typedef typename type_of<RetTypeID>::type ret_type;

      void single(char *ret, char *const *args)
      {
        const char *begin = *reinterpret_cast<const char **>(args[0]);
        const char *end = *reinterpret_cast<const char **>(args[1]);

        if (!_parse_number(begin, end)) {
          throw std::runtime_error("JSON error");
        }

        *reinterpret_cast<ret_type *>(ret) = dynd::parse<ret_type>(begin, end);
        *reinterpret_cast<const char **>(args[0]) = end;
      }
    };

    template <>
    struct parse_kernel<bool_type_id> : base_kernel<parse_kernel<bool_type_id>, 2> {
      void single(char *ret, char *const *args)
      {
        *reinterpret_cast<bool1 *>(ret) =
            parse_bool(*reinterpret_cast<const char **>(args[0]), *reinterpret_cast<const char **>(args[1]));
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
        const char *begin = *reinterpret_cast<const char **>(args[0]);
        const char *end = *reinterpret_cast<const char **>(args[1]);

        if (_parse_na(begin, end)) {
          get_child()->single(ret, nullptr);
          *reinterpret_cast<const char **>(args[0]) = begin;
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

    /*
    static bool parse_struct_json_from_object(const ndt::type &tp, const char *arrmeta, char *out_data, const char
    *&begin,
                                              const char *end, const eval::eval_context *ectx)
    {
      const char *saved_begin = begin;
      if (!parse_token(begin, end, "{")) {
        return false;
      }

      const ndt::struct_type *fsd = tp.extended<ndt::struct_type>();
      intptr_t field_count = fsd->get_field_count();
      const size_t *data_offsets = fsd->get_data_offsets(arrmeta);
      const size_t *arrmeta_offsets = fsd->get_arrmeta_offsets_raw();

      // Keep track of which fields we've seen
      shortvector<bool> populated_fields(field_count);
      memset(populated_fields.get(), 0, sizeof(bool) * field_count);

      // If it's not an empty object, start the loop parsing the elements
      if (!parse_token(begin, end, "}")) {
        for (;;) {
          const char *strbegin, *strend;
          bool escaped;
          skip_whitespace(begin, end);
          if (!parse_doublequote_string_no_ws(begin, end, strbegin, strend, escaped)) {
            throw json_parse_error(begin, "expected string for name in object dict", tp);
          }
          if (!parse_token(begin, end, ":")) {
            throw json_parse_error(begin, "expected ':' separating name from value in object dict", tp);
          }
          intptr_t i;
          if (escaped) {
            std::string name;
            unescape_string(strbegin, strend, name);
            i = fsd->get_field_index(name);
          }
          else {
            i = fsd->get_field_index(strbegin, strend);
          }
          if (i == -1) {
            // TODO: Add an error policy to this parser of whether to throw an error
            //       or not. For now, just throw away fields not in the destination.
            skip_json_value(begin, end);
          }
          else {
            parse_json(fsd->get_field_type(i), arrmeta + arrmeta_offsets[i], out_data + data_offsets[i], begin, end,
    ectx);
            populated_fields[i] = true;
          }
          if (!parse_token(begin, end, ",")) {
            break;
          }
        }
        if (!parse_token(begin, end, "}")) {
          throw json_parse_error(begin, "expected object dict separator ',' or terminator '}'", tp);
        }
      }

      for (intptr_t i = 0; i < field_count; ++i) {
        if (!populated_fields[i]) {
          const ndt::type &field_tp = fsd->get_field_type(i);
          if (field_tp.get_type_id() == option_type_id) {
            field_tp.extended<ndt::option_type>()->assign_na(arrmeta + arrmeta_offsets[i], out_data + data_offsets[i],
                                                             &eval::default_eval_context);
          }
          else {
            stringstream ss;
            ss << "object dict does not contain the field ";
            print_escaped_utf8_string(ss, fsd->get_field_name(i));
            ss << " as required by the data type";
            skip_whitespace(saved_begin, end);
            throw json_parse_error(saved_begin, ss.str(), tp);
          }
        }
      }

      return true;
    }
    */

    template <>
    struct parse_kernel<struct_type_id> : base_kernel<parse_kernel<struct_type_id>, 2> {
      ndt::type res_tp;
      size_t field_count;
      const size_t *data_offsets;
      std::vector<intptr_t> child_offsets;

      parse_kernel(const ndt::type &res_tp, size_t field_count, const size_t *data_offsets)
          : res_tp(res_tp), field_count(field_count), data_offsets(data_offsets), child_offsets(field_count)
      {
      }

      ~parse_kernel()
      {
        for (intptr_t offset : child_offsets) {
          get_child(offset)->destroy();
        }
      }

      void single(char *res, char *const *args)
      {

        const char *&begin = *reinterpret_cast<const char **>(args[0]);
        const char *&end = *reinterpret_cast<const char **>(args[1]);

        //        const char *saved_begin = *reinterpret_cast<const char **>(args[0]);
        if (!parse_token(args, "{")) {
          throw json_parse_error(args, "expected object dict starting with '{'", res_tp);
        }

        shortvector<bool> populated_fields(field_count);
        memset(populated_fields.get(), 0, sizeof(bool) * field_count);

        if (!parse_token(args, "}")) {
          for (;;) {
            const char *strbegin, *strend;
            bool escaped;
            skip_whitespace(args);
            if (!parse_doublequote_string_no_ws(begin, end, strbegin, strend, escaped)) {
              throw json_parse_error(args, "expected string for name in object dict", res_tp);
            }
            if (!parse_token(args, ":")) {
              throw json_parse_error(args, "expected ':' separating name from value in object dict", res_tp);
            }
            intptr_t i;
            if (escaped) {
              std::string name;
              unescape_string(strbegin, strend, name);
              i = res_tp.extended<ndt::struct_type>()->get_field_index(name);
            }
            else {
              i = res_tp.extended<ndt::struct_type>()->get_field_index(strbegin, strend);
            }

            get_child(child_offsets[i])->single(res + data_offsets[i], args);
            populated_fields[i] = true;
            if (!parse_token(args, ",")) {
              break;
            }
          }
        }

        if (!parse_token(args, "}")) {
          throw json_parse_error(args, "expected object dict separator ',' or terminator '}'", res_tp);
        }
      }

      static intptr_t instantiate(char *DYND_UNUSED(static_data), char *data, void *ckb, intptr_t ckb_offset,
                                  const ndt::type &dst_tp, const char *dst_arrmeta, intptr_t nsrc,
                                  const ndt::type *src_tp, const char *const *src_arrmeta, kernel_request_t kernreq,
                                  const eval::eval_context *DYND_UNUSED(ectx), intptr_t nkwd, const nd::array *kwds,
                                  const std::map<std::string, ndt::type> &tp_vars)
      {
        size_t field_count = dst_tp.extended<ndt::struct_type>()->get_field_count();
        const size_t *arrmeta_offsets = dst_tp.extended<ndt::struct_type>()->get_arrmeta_offsets_raw();

        intptr_t self_offset = ckb_offset;
        make(ckb, kernreq, ckb_offset, dst_tp, field_count,
             dst_tp.extended<ndt::struct_type>()->get_data_offsets(dst_arrmeta));

        for (size_t i = 0; i < field_count; ++i) {
          get_self(reinterpret_cast<ckernel_builder<kernel_request_host> *>(ckb), self_offset)->child_offsets[i] =
              ckb_offset - self_offset;
          ckb_offset = json::parse::get()->instantiate(json::parse::get()->static_data(), data, ckb, ckb_offset,
                                                       dst_tp.extended<ndt::struct_type>()->get_field_type(i),
                                                       dst_arrmeta + arrmeta_offsets[i], nsrc, src_tp, src_arrmeta,
                                                       kernreq, NULL, nkwd, kwds, tp_vars);
        }

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

    template <>
    struct parse_kernel<var_dim_type_id> : base_kernel<parse_kernel<var_dim_type_id>> {
      typedef ndt::var_dim_type::data_type ret_type;

      ndt::type ret_tp;
      intrusive_ptr<memory_block_data> blockref;
      intptr_t stride;

      parse_kernel(const ndt::type &ret_tp, const intrusive_ptr<memory_block_data> &blockref, intptr_t stride)
          : ret_tp(ret_tp), blockref(blockref), stride(stride)
      {
      }

      ~parse_kernel() { get_child()->destroy(); }

      void single(char *ret, char *const *args)
      {
        if (!parse_token(args, "[")) {
          throw json_parse_error(args, "expected list starting with '['", ret_tp);
        }
        skip_whitespace(args);

        memory_block_data::api *allocator = blockref->get_api();
        size_t size = 0, allocated_size = 8;
        reinterpret_cast<ret_type *>(ret)->begin = allocator->allocate(blockref.get(), allocated_size);

        ckernel_prefix *child = get_child();
        for (char *data = reinterpret_cast<ret_type *>(ret)->begin;; data += stride) {
          // Increase the allocated array size if necessary
          if (size == allocated_size) {
            allocated_size *= 2;
            reinterpret_cast<ret_type *>(ret)->begin =
                allocator->resize(blockref.get(), reinterpret_cast<ret_type *>(ret)->begin, allocated_size);
          }
          ++size;
          reinterpret_cast<ndt::var_dim_type::data_type *>(ret)->size = size;

          child->single(data, args);

          if (!parse_token(args, ",")) {
            break;
          }
          skip_whitespace(args);
        }

        if (!parse_token(args, "]")) {
          throw json_parse_error(args, "array is too long, expected list terminator ']'", ret_tp);
        }

        // Shrink-wrap the memory to just fit the string
        reinterpret_cast<ret_type *>(ret)->begin =
            allocator->resize(blockref.get(), reinterpret_cast<ret_type *>(ret)->begin, size);
        reinterpret_cast<ret_type *>(ret)->size = size;
      }

      static intptr_t instantiate(char *DYND_UNUSED(static_data), char *data, void *ckb, intptr_t ckb_offset,
                                  const ndt::type &dst_tp, const char *dst_arrmeta, intptr_t nsrc,
                                  const ndt::type *src_tp, const char *const *src_arrmeta, kernel_request_t kernreq,
                                  const eval::eval_context *DYND_UNUSED(ectx), intptr_t nkwd, const nd::array *kwds,
                                  const std::map<std::string, ndt::type> &tp_vars)
      {
        make(ckb, kernreq, ckb_offset, dst_tp,
             reinterpret_cast<const ndt::var_dim_type::metadata_type *>(dst_arrmeta)->blockref,
             reinterpret_cast<const ndt::var_dim_type::metadata_type *>(dst_arrmeta)->stride);

        const ndt::type &child_dst_tp = dst_tp.extended<ndt::var_dim_type>()->get_element_type();
        return json::parse::get()->instantiate(json::parse::get()->static_data(), data, ckb, ckb_offset, child_dst_tp,
                                               dst_arrmeta + sizeof(ndt::var_dim_type::metadata_type), nsrc, src_tp,
                                               src_arrmeta, kernreq, NULL, nkwd, kwds, tp_vars);
      }
    };

  } // namespace dynd::nd::json
} // namespace dynd::nd

namespace ndt {

  template <type_id_t DstTypeID>
  struct traits<nd::json::parse_kernel<DstTypeID>> {
    static type equivalent() { return callable_type::make(DstTypeID, {make_type<char *>(), make_type<char *>()}); }
  };

} // namespace dynd::ndt
} // namespace dynd
