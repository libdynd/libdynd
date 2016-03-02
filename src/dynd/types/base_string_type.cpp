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

std::map<std::string, std::pair<ndt::type, void *>> ndt::base_string_type::get_dynamic_type_properties() const
{
  std::map<std::string, std::pair<ndt::type, void *>> properties;
  //properties["encoding"] = {ndt::type("uint64"), get_encoding()};

  return properties;
}
