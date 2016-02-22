//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/callable.hpp>
#include <dynd/type.hpp>
#include <dynd/shape_tools.hpp>

using namespace std;
using namespace dynd;

ndt::base_string_type::~base_string_type() {}

std::string ndt::base_string_type::get_utf8_string(const char *arrmeta, const char *data,
                                                   assign_error_mode errmode) const
{
  const char *begin, *end;
  get_string_range(&begin, &end, arrmeta, data);
  return string_range_as_utf8_string(get_encoding(), begin, end, errmode);
}

size_t ndt::base_string_type::get_iterdata_size(intptr_t DYND_UNUSED(ndim)) const { return 0; }

static void get_extended_string_encoding(const ndt::type &dt)
{
  const ndt::base_string_type *d = dt.extended<ndt::base_string_type>();
  stringstream ss;
  ss << d->get_encoding();
  //  return ss.str();
}

static const std::map<std::string, nd::callable> &base_string_type_properties()
{
  static const std::map<std::string, nd::callable> base_string_type_properties{
      {"encoding", nd::functional::apply(&get_extended_string_encoding)}};

  return base_string_type_properties;
}

std::map<std::string, nd::callable> ndt::base_string_type::get_dynamic_type_properties() const
{
  return base_string_type_properties();
}


