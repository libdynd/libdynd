//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/kernels/base_kernel.hpp>
#include <dynd/option.hpp>
#include <dynd/parse.hpp>

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
    struct parse_kernel : base_strided_kernel<parse_kernel<RetTypeID>, 2> {
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
    struct parse_kernel<bool_id> : base_strided_kernel<parse_kernel<bool_id>, 2> {
      void single(char *ret, char *const *args)
      {
        *reinterpret_cast<bool1 *>(ret) =
            parse_bool(*reinterpret_cast<const char **>(args[0]), *reinterpret_cast<const char **>(args[1]));
      }
    };

    template <>
    struct parse_kernel<string_id> : base_strided_kernel<parse_kernel<string_id>, 2> {
      void single(char *res, char *const *args)
      {
        const char *&rbegin = *reinterpret_cast<const char **>(args[0]);
        const char *begin = *reinterpret_cast<const char **>(args[0]);
        const char *end = *reinterpret_cast<const char **>(args[1]);

        skip_whitespace(begin, end);
        const char *strbegin, *strend;
        bool escaped;

        if (parse_doublequote_string_no_ws(begin, end, strbegin, strend, escaped)) {
          std::string val;
          unescape_string(strbegin, strend, val);
          reinterpret_cast<string *>(res)->assign(strbegin, strend - strbegin);

          /*
                    try {
                      if (!escaped) {
                        bsd->set_from_utf8_string(arrmeta, out_data, strbegin, strend, ectx);
                      }
                      else {
                        std::string val;
                        unescape_string(strbegin, strend, val);
                        bsd->set_from_utf8_string(arrmeta, out_data, val, ectx);
                      }
                    }
                    catch (const std::exception &e) {
                      skip_whitespace(rbegin, begin);
                      throw json_parse_error(rbegin, e.what(), tp);
                    }
                    catch (const dynd::dynd_exception &e) {
                      skip_whitespace(rbegin, begin);
                      throw json_parse_error(rbegin, e.what(), tp);
                    }
          */
        }
        else {
          throw json_parse_error(begin, "expected a string", ndt::type());
        }
        rbegin = begin;
      }
    };

    /*
    static void parse_string_json(const ndt::type &tp, const char *arrmeta, char *out_data, const char *&rbegin,
                                  const char *end, const eval::eval_context *ectx)
    {
      const char *begin = rbegin;
      skip_whitespace(begin, end);
      const char *strbegin, *strend;
      bool escaped;
      if (parse_doublequote_string_no_ws(begin, end, strbegin, strend, escaped)) {
        const ndt::base_string_type *bsd = tp.extended<ndt::base_string_type>();
        try {
          if (!escaped) {
            bsd->set_from_utf8_string(arrmeta, out_data, strbegin, strend, ectx);
          }
          else {
            std::string val;
            unescape_string(strbegin, strend, val);
            bsd->set_from_utf8_string(arrmeta, out_data, val, ectx);
          }
        }
        catch (const std::exception &e) {
          skip_whitespace(rbegin, begin);
          throw json_parse_error(rbegin, e.what(), tp);
        }
        catch (const dynd::dynd_exception &e) {
          skip_whitespace(rbegin, begin);
          throw json_parse_error(rbegin, e.what(), tp);
        }
      }
      else {
        throw json_parse_error(begin, "expected a string", tp);
      }
      rbegin = begin;
    }
    */

    template <>
    struct parse_kernel<option_id> : base_strided_kernel<parse_kernel<option_id>, 2> {
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

      static void instantiate(char *DYND_UNUSED(static_data), char *data, kernel_builder *ckb, const ndt::type &dst_tp,
                              const char *dst_arrmeta, intptr_t nsrc, const ndt::type *src_tp,
                              const char *const *src_arrmeta, kernel_request_t kernreq, intptr_t nkwd,
                              const nd::array *kwds, const std::map<std::string, ndt::type> &tp_vars)
      {
        intptr_t ckb_offset = ckb->size();
        intptr_t self_offset = ckb_offset;
        ckb->emplace_back<parse_kernel>(kernreq);
        ckb_offset = ckb->size();

        assign_na::get()->instantiate(assign_na::get()->static_data(), data, ckb, dst_tp, dst_arrmeta, 0, nullptr,
                                      nullptr, kernreq | kernel_request_data_only, nkwd, kwds, tp_vars);
        ckb_offset = ckb->size();

        ckb->get_at<parse_kernel>(self_offset)->parse_offset = ckb_offset - self_offset;
        parse::get()->instantiate(parse::get()->static_data(), data, ckb,
                                  dst_tp.extended<ndt::option_type>()->get_value_type(), dst_arrmeta, nsrc, src_tp,
                                  src_arrmeta, kernreq | kernel_request_data_only, nkwd, kwds, tp_vars);
        ckb_offset = ckb->size();
      }
    };

    template <>
    struct parse_kernel<struct_id> : base_strided_kernel<parse_kernel<struct_id>, 2> {
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
              i = res_tp.extended<ndt::struct_type>()->get_field_index(std::string(strbegin, strend));
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

      static void instantiate(char *DYND_UNUSED(static_data), char *data, kernel_builder *ckb, const ndt::type &dst_tp,
                              const char *dst_arrmeta, intptr_t nsrc, const ndt::type *src_tp,
                              const char *const *src_arrmeta, kernel_request_t kernreq, intptr_t nkwd,
                              const nd::array *kwds, const std::map<std::string, ndt::type> &tp_vars)
      {
        intptr_t ckb_offset = ckb->size();
        size_t field_count = dst_tp.extended<ndt::struct_type>()->get_field_count();
        const std::vector<uintptr_t> &arrmeta_offsets = dst_tp.extended<ndt::struct_type>()->get_arrmeta_offsets();

        intptr_t self_offset = ckb_offset;
        ckb->emplace_back<parse_kernel>(kernreq, dst_tp, field_count,
                                        dst_tp.extended<ndt::struct_type>()->get_data_offsets(dst_arrmeta));
        ckb_offset = ckb->size();

        for (size_t i = 0; i < field_count; ++i) {
          ckb->get_at<parse_kernel>(self_offset)->child_offsets[i] = ckb_offset - self_offset;
          json::parse::get()->instantiate(
              json::parse::get()->static_data(), data, ckb, dst_tp.extended<ndt::struct_type>()->get_field_type(i),
              dst_arrmeta + arrmeta_offsets[i], nsrc, src_tp, src_arrmeta, kernreq, nkwd, kwds, tp_vars);
          ckb_offset = ckb->size();
        }
      }
    };

    template <>
    struct parse_kernel<fixed_dim_id> : base_strided_kernel<parse_kernel<fixed_dim_id>, 2> {
      ndt::type ret_tp;
      size_t _size;
      intptr_t stride;

      ~parse_kernel() { get_child()->destroy(); }

      parse_kernel(const ndt::type &ret_tp, size_t size, intptr_t stride) : ret_tp(ret_tp), _size(size), stride(stride)
      {
      }

      void single(char *ret, char *const *args)
      {
        if (!parse_token(args, "[")) {
          throw json_parse_error(args, "expected list starting with '['", ret_tp);
        }
        skip_whitespace(args);

        kernel_prefix *child = get_child();
        for (size_t i = 0; i < _size; ++i) {
          child->single(ret, args);
          if (i < _size - 1 && !parse_token(args, ",")) {
            throw json_parse_error(args, "array is too short, expected ',' list item separator", ret_tp);
          }
          skip_whitespace(args);

          ret += stride;
        }

        if (!parse_token(args, "]")) {
          throw json_parse_error(args, "array is too long, expected list terminator ']'", ret_tp);
        }
      }

      static void instantiate(char *DYND_UNUSED(static_data), char *data, kernel_builder *ckb, const ndt::type &dst_tp,
                              const char *dst_arrmeta, intptr_t nsrc, const ndt::type *src_tp,
                              const char *const *src_arrmeta, kernel_request_t kernreq, intptr_t nkwd,
                              const nd::array *kwds, const std::map<std::string, ndt::type> &tp_vars)
      {
        ckb->emplace_back<parse_kernel>(kernreq, dst_tp, reinterpret_cast<const size_stride_t *>(dst_arrmeta)->dim_size,
                                        reinterpret_cast<const size_stride_t *>(dst_arrmeta)->stride);

        const ndt::type &child_dst_tp = dst_tp.extended<ndt::fixed_dim_type>()->get_element_type();
        json::parse::get()->instantiate(json::parse::get()->static_data(), data, ckb, child_dst_tp,
                                        dst_arrmeta + sizeof(ndt::fixed_dim_type::metadata_type), nsrc, src_tp,
                                        src_arrmeta, kernreq, nkwd, kwds, tp_vars);
      }
    };

    template <>
    struct parse_kernel<var_dim_id> : base_strided_kernel<parse_kernel<var_dim_id>, 2> {
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

        size_t size = 0, allocated_size = 8;
        reinterpret_cast<ret_type *>(ret)->begin = blockref->alloc(allocated_size);

        kernel_prefix *child = get_child();
        for (char *data = reinterpret_cast<ret_type *>(ret)->begin;; data += stride) {
          // Increase the allocated array size if necessary
          if (size == allocated_size) {
            allocated_size *= 2;
            reinterpret_cast<ret_type *>(ret)->begin =
                blockref->resize(reinterpret_cast<ret_type *>(ret)->begin, allocated_size);
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
        reinterpret_cast<ret_type *>(ret)->begin = blockref->resize(reinterpret_cast<ret_type *>(ret)->begin, size);
        reinterpret_cast<ret_type *>(ret)->size = size;
      }

      static void instantiate(char *DYND_UNUSED(static_data), char *data, kernel_builder *ckb, const ndt::type &dst_tp,
                              const char *dst_arrmeta, intptr_t nsrc, const ndt::type *src_tp,
                              const char *const *src_arrmeta, kernel_request_t kernreq, intptr_t nkwd,
                              const nd::array *kwds, const std::map<std::string, ndt::type> &tp_vars)
      {
        ckb->emplace_back<parse_kernel>(
            kernreq, dst_tp, reinterpret_cast<const ndt::var_dim_type::metadata_type *>(dst_arrmeta)->blockref,
            reinterpret_cast<const ndt::var_dim_type::metadata_type *>(dst_arrmeta)->stride);

        const ndt::type &child_dst_tp = dst_tp.extended<ndt::var_dim_type>()->get_element_type();
        json::parse::get()->instantiate(json::parse::get()->static_data(), data, ckb, child_dst_tp,
                                        dst_arrmeta + sizeof(ndt::var_dim_type::metadata_type), nsrc, src_tp,
                                        src_arrmeta, kernreq, nkwd, kwds, tp_vars);
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
