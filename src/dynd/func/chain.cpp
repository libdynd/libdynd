//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/func/chain.hpp>
#include <dynd/kernels/chain_kernel.hpp>

using namespace std;
using namespace dynd;

nd::arrfunc nd::functional::chain(const nd::arrfunc &first,
                                  const nd::arrfunc &second,
                                  const ndt::type &buf_tp)
{
  if (first.get_type()->get_npos() != 1) {
    throw runtime_error("Multi-parameter arrfunc chaining is not implemented");
  }

  if (second.get_type()->get_npos() != 1) {
    stringstream ss;
    ss << "Cannot chain functions " << first << " and " << second
       << ", because the second function is not unary";
    throw invalid_argument(ss.str());
  }

  if (buf_tp.get_type_id() == uninitialized_type_id) {
    throw runtime_error("Chaining functions without a provided intermediate "
                        "type is not implemented");
  }

  return as_arrfunc<chain_kernel>(
      ndt::make_arrfunc(first.get_type()->get_pos_tuple(),
                        second.get_type()->get_return_type()),
      chain_kernel::static_data(first, second, buf_tp),
      first.get()->data_size + second.get()->data_size);
}
