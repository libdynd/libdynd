//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/types/ndarrayarg_type.hpp>
#include <dynd/kernels/assignment_kernels.hpp>

using namespace std;
using namespace dynd;

void ndt::ndarrayarg_type::print_data(std::ostream &o, const char *DYND_UNUSED(arrmeta), const char *data) const
{
  o << *reinterpret_cast<const nd::array *>(data);
}

void ndt::ndarrayarg_type::print_type(std::ostream &o) const { o << "ndarrayarg"; }

bool ndt::ndarrayarg_type::is_lossless_assignment(const type &dst_tp, const type &src_tp) const
{
  if (dst_tp.extended() == this) {
    if (src_tp.extended() == this) {
      return true;
    }
    else {
      return src_tp.get_type_id() == ndarrayarg_type_id;
    }
  }
  else {
    return false;
  }
}

bool ndt::ndarrayarg_type::operator==(const base_type &rhs) const { return rhs.get_type_id() == ndarrayarg_type_id; }

size_t ndt::ndarrayarg_type::make_comparison_kernel(void *DYND_UNUSED(ckb), intptr_t DYND_UNUSED(ckb_offset),
                                                    const type &src0_tp, const char *DYND_UNUSED(src0_arrmeta),
                                                    const type &src1_tp, const char *DYND_UNUSED(src1_arrmeta),
                                                    comparison_type_t comptype,
                                                    const eval::eval_context *DYND_UNUSED(ectx)) const
{
  throw not_comparable_error(src0_tp, src1_tp, comptype);
}
