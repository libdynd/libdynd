//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/func/chain.hpp>
#include <dynd/kernels/chain.hpp>

using namespace std;
using namespace dynd;

nd::arrfunc nd::functional::chain(const nd::arrfunc &first,
                                  const nd::arrfunc &second,
                                  const ndt::type &buf_tp)
{
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

  instantiate_chain_data icd;
  icd.first = first;
  icd.second = second;
  icd.buf_tp = buf_tp;

  return as_arrfunc<unary_heap_chain_ck>(
      ndt::make_arrfunc(first.get_type()->get_pos_tuple(),
                        second.get_type()->get_return_type()),
      icd, first.get()->data_size + second.get()->data_size);
}
