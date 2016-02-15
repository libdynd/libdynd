//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/kernels/kernel_prefix.hpp>
#include <dynd/types/substitute_typevars.hpp>

using namespace std;
using namespace dynd;

void nd::kernel_prefix::resolve_dst_type(char *DYND_UNUSED(static_data), char *DYND_UNUSED(data), ndt::type &dst_tp,
                                         intptr_t DYND_UNUSED(nsrc), const ndt::type *DYND_UNUSED(src_tp),
                                         intptr_t DYND_UNUSED(nkwd), const array *DYND_UNUSED(kwds),
                                         const std::map<std::string, ndt::type> &tp_vars)
{
  dst_tp = ndt::substitute(dst_tp, tp_vars, true);
}
