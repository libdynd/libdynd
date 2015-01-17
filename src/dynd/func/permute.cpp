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

nd::arrfunc nd::functional::permute(const arrfunc &child,
                                    const std::vector<intptr_t> &perm)
{
  const arrfunc_type *child_tp = child.get_type();

  ndt::type ret_tp;
  intptr_t npos = perm.size();
  std::vector<ndt::type> pos_tp(npos);
  for (size_t i = 0; i < perm.size(); ++i) {
    intptr_t j = perm[i];

    if (j < -1 || j >= child_tp->get_npos()) {
    }

    if (j == -1) {
      if (child_tp->get_return_type().get_type_id() != void_type_id) {
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

  ndt::type self_tp =
      ndt::make_arrfunc(ndt::make_tuple(nd::array(pos_tp.data(), npos)),
                        child_tp->get_kwd_struct(), ret_tp);
  return as_arrfunc<kernels::permute_ck<2>>(
      self_tp, std::make_pair(child, perm));
}