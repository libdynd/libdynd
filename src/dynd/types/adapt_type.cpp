//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/types/adapt_type.hpp>

using namespace std;
using namespace dynd;

ndt::adapt_type::adapt_type(const type &operand_type, const type &value_type, const std::string &DYND_UNUSED(op))
    : base_expr_type(adapt_type_id, expr_kind, operand_type.get_data_size(), operand_type.get_data_alignment(),
                     inherited_flags(value_type.get_flags(), operand_type.get_flags()), 0)
{
}

void ndt::adapt_type::print_data(std::ostream &DYND_UNUSED(o), const char *DYND_UNUSED(arrmeta),
                                 const char *DYND_UNUSED(data)) const
{
  throw std::runtime_error("not implemented");
}

void ndt::adapt_type::print_type(std::ostream &DYND_UNUSED(o)) const { throw std::runtime_error("not implemented"); }

bool ndt::adapt_type::is_lossless_assignment(const type &DYND_UNUSED(dst_tp), const type &DYND_UNUSED(src_tp)) const
{
  throw std::runtime_error("not implemented");
}

bool ndt::adapt_type::operator==(const base_type &DYND_UNUSED(rhs)) const
{
  throw std::runtime_error("not implemented");
}

ndt::type ndt::adapt_type::with_replaced_storage_type(const type &DYND_UNUSED(replacement_type)) const
{
  throw std::runtime_error("not implemented");
}
