//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/func/arrfunc.hpp>
#include <dynd/func/call.hpp>
#include <dynd/func/elwise.hpp>
#include <dynd/func/multidispatch.hpp>
#include <dynd/kernels/compare_kernels.hpp>

namespace dynd {
namespace nd {

  template <typename F, template <type_id_t...> class K, int N>
  struct comparison_operator;

  template <typename F, template <type_id_t, type_id_t> class K>
  struct comparison_operator<F, K, 2> : declfunc<F> {
    static arrfunc children[DYND_TYPE_ID_MAX + 1][DYND_TYPE_ID_MAX + 1];
    static arrfunc default_child;

    static arrfunc make()
    {
      typedef type_id_sequence<int8_type_id, int16_type_id, int32_type_id,
                               int64_type_id, float32_type_id,
                               float64_type_id> numeric_type_ids;

      arrfunc self = functional::call<F>(ndt::type("(Any, Any) -> Any"));

      for (const std::pair<std::pair<type_id_t, type_id_t>, arrfunc> &pair :
           arrfunc::make_all<K, numeric_type_ids, numeric_type_ids>()) {
        children[pair.first.first][pair.first.second] = pair.second;
      }

      for (type_id_t i0 : numeric_type_ids::vals()) {
        for (type_id_t i1 : dim_type_ids::vals()) {
          const ndt::type child_tp = ndt::arrfunc_type::make(
              {ndt::type(i0), ndt::type(i1)}, ndt::type("Any"));
          children[i0][i1] = functional::elwise(child_tp, self);
        }
      }

      for (type_id_t i0 : dim_type_ids::vals()) {
        typedef join<numeric_type_ids, dim_type_ids>::type type_ids;
        for (type_id_t i1 : type_ids::vals()) {
          const ndt::type child_tp = ndt::arrfunc_type::make(
              {ndt::type(i0), ndt::type(i1)}, ndt::type("Any"));
          children[i0][i1] = functional::elwise(child_tp, self);
        }
      }

      return functional::multidispatch_by_type_id(self.get_array_type(),
                                                  children, default_child);
    }
  };

  template <typename T, template <type_id_t, type_id_t> class K>
  arrfunc comparison_operator<T, K, 2>::children[DYND_TYPE_ID_MAX +
                                                 1][DYND_TYPE_ID_MAX + 1];

  template <typename T, template <type_id_t, type_id_t> class K>
  arrfunc comparison_operator<T, K, 2>::default_child;

  extern struct less : comparison_operator<less, less_kernel, 2> {
  } less;

  extern struct less_equal
      : comparison_operator<less_equal, less_equal_kernel, 2> {
  } less_equal;

  extern struct equal : comparison_operator<equal, equal_kernel, 2> {
  } equal;

  extern struct not_equal
      : comparison_operator<not_equal, not_equal_kernel, 2> {
  } not_equal;

  extern struct greater_equal
      : comparison_operator<greater_equal, greater_equal_kernel, 2> {
  } greater_equal;

  extern struct greater : comparison_operator<greater, greater_kernel, 2> {
  } greater;

} // namespace dynd::nd
} // namespace dynd