//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/search.hpp>
#include <dynd/kernels/binary_search_kernel.hpp>
#include <dynd/types/fixed_dim_type.hpp>

using namespace std;
using namespace dynd;

intptr_t nd::binary_search_old(const nd::array &n, const char *arrmeta, const char *data)
{
  const char *n_arrmeta = n.get_arrmeta();
  ndt::type element_tp = n.get_type().at_single(0, &n_arrmeta);
  if (element_tp.get_arrmeta_size() == 0 || n_arrmeta == arrmeta ||
      memcmp(n_arrmeta, arrmeta, element_tp.get_arrmeta_size()) == 0) {
    // First, a version where the arrmeta is identical, so we can
    // make do with only a single comparison kernel
    ckernel_builder<kernel_request_host> k_n_less_d;
    make_comparison_kernel(&k_n_less_d, 0, element_tp, n_arrmeta, element_tp, n_arrmeta, comparison_type_sorting_less,
                           &eval::default_eval_context);
    expr_single_t fn_n_less_d = k_n_less_d.get()->get_function<expr_single_t>();

    // TODO: support any type of array dimension
    if (n.get_type().get_type_id() != fixed_dim_type_id) {
      stringstream ss;
      ss << "TODO: binary_search on array with type " << n.get_type() << " is not implemented";
      throw runtime_error(ss.str());
    }

    const char *n_data = n.get_readonly_originptr();
    intptr_t n_stride = reinterpret_cast<const fixed_dim_type_arrmeta *>(n.get_arrmeta())->stride;
    intptr_t first = 0, last = n.get_dim_size();
    while (first < last) {
      intptr_t trial = first + (last - first) / 2;
      const char *trial_data = n_data + trial * n_stride;

      // In order for the data to always match up with the arrmeta, need to have
      // trial_data first and data second in the comparison operations.
      const char *const src_try0[2] = {data, trial_data};
      const char *const src_try1[2] = {trial_data, data};
      int dst;
      fn_n_less_d(k_n_less_d.get(), reinterpret_cast<char *>(&dst), const_cast<char *const *>(src_try0));
      if (dst) {
        // value < arr[trial]
        last = trial;
      } else {
        int dst;
        fn_n_less_d(k_n_less_d.get(), reinterpret_cast<char *>(&dst), const_cast<char *const *>(src_try1));
        if (dst) {
          // value > arr[trial]
          first = trial + 1;
        } else {
          return trial;
        }
      }
    }
    return -1;
  } else {
    // Second, a version where the arrmeta are different, so
    // we need to get a kernel for each comparison direction.
    ckernel_builder<kernel_request_host> k_n_less_d;
    make_comparison_kernel(&k_n_less_d, 0, element_tp, n_arrmeta, element_tp, arrmeta, comparison_type_sorting_less,
                           &eval::default_eval_context);
    expr_single_t f_n_less_d = k_n_less_d.get()->get_function<expr_single_t>();

    ckernel_builder<kernel_request_host> k_d_less_n;
    make_comparison_kernel(&k_d_less_n, 0, element_tp, arrmeta, element_tp, n_arrmeta, comparison_type_sorting_less,
                           &eval::default_eval_context);
    expr_single_t f_d_less_n = k_d_less_n.get()->get_function<expr_single_t>();

    // TODO: support any type of array dimension
    if (n.get_type().get_type_id() != fixed_dim_type_id) {
      stringstream ss;
      ss << "TODO: binary_search on array with type " << n.get_type() << " is not implemented";
      throw runtime_error(ss.str());
    }

    const char *n_data = n.get_readonly_originptr();
    intptr_t n_stride = reinterpret_cast<const fixed_dim_type_arrmeta *>(n.get_arrmeta())->stride;
    intptr_t first = 0, last = n.get_dim_size();
    while (first < last) {
      intptr_t trial = first + (last - first) / 2;
      const char *trial_data = n_data + trial * n_stride;

      // In order for the data to always match up with the arrmeta, need to have
      // trial_data first and data second in the comparison operations.
      const char *const src_try0[2] = {data, trial_data};
      const char *const src_try1[2] = {trial_data, data};
      int dst;
      f_d_less_n(k_d_less_n.get(), reinterpret_cast<char *>(&dst), const_cast<char *const *>(src_try0));
      if (dst) {
        // value < arr[trial]
        last = trial;
      } else {
        int dst;
        f_n_less_d(k_n_less_d.get(), reinterpret_cast<char *>(&dst), const_cast<char *const *>(src_try1));
        if (dst) {
          // value > arr[trial]
          first = trial + 1;
        } else {
          return trial;
        }
      }
    }
    return -1;
  }
}

DYND_API nd::callable nd::binary_search::make()
{
  return callable::make<binary_search_kernel>();
}

DYND_API struct nd::binary_search nd::binary_search;
