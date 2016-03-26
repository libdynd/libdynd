//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <stdexcept>
#include <string>

#include <dynd/callable.hpp>
#include <dynd/config.hpp>
#include <dynd/string_encodings.hpp>
#include <dynd/type.hpp>
#include <dynd/types/string_type.hpp>
#include <dynd/types/option_type.hpp>
#include <dynd/option.hpp>
#include <dynd/parse_util.hpp>

namespace dynd {
namespace nd {
  namespace json {

    extern DYND_API callable dynamic_parse;

    inline array parse(const ndt::type &ret_tp, const char *begin, const char *end)
    {
      skip_whitespace(begin, end);

      ndt::type dst_tp2 = ret_tp;
      char *args_data[2] = {reinterpret_cast<char *>(&begin), reinterpret_cast<char *>(&end)};
      nd::array ret =
          dynamic_parse->call(dst_tp2, 0, nullptr, nullptr, args_data, 0, nullptr, std::map<std::string, ndt::type>());

      skip_whitespace(begin, end);
      if (begin != end) {
        throw json_parse_error(begin, "unexpected trailing JSON text", ret_tp);
      }

      return ret;
    }

    inline array parse(const ndt::type &ret_tp, const char *begin)
    {
      return parse(ret_tp, begin, begin + std::strlen(begin));
    }

    inline array parse(const ndt::type &ret_tp, const std::string &s)
    {
      return parse(ret_tp, s.c_str(), s.c_str() + s.size());
    }

  } // namespace dynd::nd::json
} // namespace dynd::nd
} // namespace dynd
