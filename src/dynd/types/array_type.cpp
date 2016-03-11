//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/type.hpp>
#include <dynd/types/array_type.hpp>

using namespace std;
using namespace dynd;

ndt::array_type::array_type()
    : base_type(array_id, sizeof(void *), alignof(void *), type_flag_construct | type_flag_destructor, 0, 0, 0)
{
}

bool ndt::array_type::operator==(const base_type &rhs) const { return this == &rhs || rhs.get_id() == array_id; }

void ndt::array_type::data_construct(const char *DYND_UNUSED(arrmeta), char *DYND_UNUSED(data)) const
{
  throw std::runtime_error("array_type::data_construct is not implemented");
}

void ndt::array_type::data_destruct(const char *DYND_UNUSED(arrmeta), char *DYND_UNUSED(data)) const
{
  throw std::runtime_error("array_type::data_destruct is not implemented");
}

void ndt::array_type::print_data(std::ostream &DYND_UNUSED(o), const char *DYND_UNUSED(arrmeta),
                                 const char *DYND_UNUSED(data)) const
{
  /*
    const nd::array &a = *reinterpret_cast<const nd::array *>(data);
    if (a.is_null()) {
      o << "null";
    }
    else {
      a.get_type().print_data(o, a.get()->metadata(), a.cdata());
    }
  */

  throw std::runtime_error("cannot print data of array_type");
}

void ndt::array_type::print_type(ostream &o) const { o << "array"; }
