//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/func/permute.hpp>

using namespace std;
using namespace dynd;

// (0, 1, 2, 3) -> (-1, 0, 1, 2)
// src_copy[i] = src[perm[i]]

// (-1, 0, 1, 2) -> (0, 1, 2, 3)
// dst
// src_copy[perm[i]] = src[i]

nd::callable nd::functional::permute(const callable &child,
                                     const std::vector<intptr_t> &perm)
{
  const ndt::callable_type *child_tp = child.get_type();

  ndt::type ret_tp;
  intptr_t npos = perm.size();
  std::vector<ndt::type> pos_tp(npos);
  for (size_t i = 0; i < perm.size(); ++i) {
    intptr_t j = perm[i];

    if (j < -1 || j >= child_tp->get_npos()) {
    }

    if (j == -1) {
      if (child_tp->get_return_type().get_id() != void_id) {
        throw std::invalid_argument("a positional argument can only be "
                                    "permuted to the return if it is "
                                    "originally void");
      }
      if (!ret_tp.is_null()) {
        throw std::invalid_argument("-1 appears twice in the permutation");
      }
      ret_tp = child_tp->get_pos_type(i);
      --npos;
    } else {
      if (!pos_tp[j].is_null()) {
        throw std::invalid_argument(std::to_string(j) +
                                    " appears twice in the permutation");
      }
      pos_tp[j] = child_tp->get_pos_type(i);
    }
  }

  if (ret_tp.is_null()) {
    ret_tp = child_tp->get_return_type();
  }

  ndt::type self_tp = ndt::callable_type::make(
      ret_tp, ndt::tuple_type::make({pos_tp.begin(), pos_tp.begin()+npos}),
      child_tp->get_kwd_struct());

  switch (child_tp->get_npos()) {
  case 2:
    return callable::make<kernels::permute_ck<2>>(
        self_tp, std::make_pair(child, perm));
  case 3:
    return callable::make<kernels::permute_ck<3>>(
        self_tp, std::make_pair(child, perm));
  case 4:
    return callable::make<kernels::permute_ck<4>>(
        self_tp, std::make_pair(child, perm));
  default:
    throw std::runtime_error("not yet implemented");
  }
}
