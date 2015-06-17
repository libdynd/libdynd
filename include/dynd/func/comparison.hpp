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

    static std::map<std::pair<type_id_t, type_id_t>, arrfunc> make_children()
    {
      typedef type_id_sequence<
          bool_type_id, int8_type_id, int16_type_id, int32_type_id,
          int64_type_id, uint8_type_id, uint16_type_id, uint32_type_id,
          uint64_type_id, float32_type_id, float64_type_id> numeric_type_ids;

      std::map<std::pair<type_id_t, type_id_t>, arrfunc> children =
          arrfunc::make_all<K, numeric_type_ids, numeric_type_ids>();

      arrfunc self = functional::call<F>(ndt::type("(Any, Any) -> Any"));

      for (type_id_t i0 : numeric_type_ids::vals()) {
        for (type_id_t i1 : dim_type_ids::vals()) {
          const ndt::type child_tp = ndt::arrfunc_type::make(
              {ndt::type(i0), ndt::type(i1)}, ndt::type("Any"));
          children[std::make_pair(i0, i1)] = functional::elwise(child_tp, self);
        }
      }

      for (type_id_t i0 : dim_type_ids::vals()) {
        typedef join<numeric_type_ids, dim_type_ids>::type type_ids;
        for (type_id_t i1 : type_ids::vals()) {
          const ndt::type child_tp = ndt::arrfunc_type::make(
              {ndt::type(i0), ndt::type(i1)}, ndt::type("Any"));
          children[std::make_pair(i0, i1)] = functional::elwise(child_tp, self);
        }
      }

      children[std::make_pair(string_type_id, string_type_id)] =
          arrfunc::make<K<string_type_id, string_type_id>>(
              ndt::type("(string, string) -> int32"));
      children[std::make_pair(fixed_string_type_id, fixed_string_type_id)] =
          arrfunc::make<K<fixed_string_type_id, fixed_string_type_id>>(
              ndt::type("(FixedString, FixedString) -> int32"));

      return children;
    }

    static arrfunc make()
    {
      for (const std::pair<std::pair<type_id_t, type_id_t>, arrfunc> &pair :
           F::make_children()) {
        children[pair.first.first][pair.first.second] = pair.second;
      }

      arrfunc child = functional::multidispatch_by_type_id(
          ndt::type("(Any, Any) -> Any"), children, default_child);

      return child;
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
    static std::map<std::pair<type_id_t, type_id_t>, arrfunc> make_children()
    {
      std::cout << "equal::make_children (start)" << std::endl;

      std::map<std::pair<type_id_t, type_id_t>, arrfunc> children =
          comparison_operator::make_children();
      children[std::make_pair(tuple_type_id, tuple_type_id)] =
          arrfunc::make<equal_kernel<tuple_type_id, tuple_type_id>>(
              ndt::type("((...), (...)) -> int32"));
      children[std::make_pair(struct_type_id, struct_type_id)] =
          arrfunc::make<equal_kernel<tuple_type_id, tuple_type_id>>(
              ndt::type("({...}, {...}) -> int32"));
      children[std::make_pair(type_type_id, type_type_id)] =
          arrfunc::make<equal_kernel<type_type_id, type_type_id>>(
              ndt::type("(type, type) -> int32"));

      std::cout << "equal::make_children (stop)" << std::endl;

      return children;
    }
  } equal;

  extern struct not_equal
      : comparison_operator<not_equal, not_equal_kernel, 2> {
    static std::map<std::pair<type_id_t, type_id_t>, arrfunc> make_children()
    {
      std::map<std::pair<type_id_t, type_id_t>, arrfunc> children =
          comparison_operator::make_children();
      children[std::make_pair(tuple_type_id, tuple_type_id)] =
          arrfunc::make<not_equal_kernel<tuple_type_id, tuple_type_id>>(
              ndt::type("((...), (...)) -> int32"));
      children[std::make_pair(struct_type_id, struct_type_id)] =
          arrfunc::make<not_equal_kernel<tuple_type_id, tuple_type_id>>(
              ndt::type("({...}, {...}) -> int32"));
      children[std::make_pair(type_type_id, type_type_id)] =
          arrfunc::make<not_equal_kernel<type_type_id, type_type_id>>(
              ndt::type("(type, type) -> int32"));

      return children;
    }
  } not_equal;

  extern struct greater_equal
      : comparison_operator<greater_equal, greater_equal_kernel, 2> {
  } greater_equal;

  extern struct greater : comparison_operator<greater, greater_kernel, 2> {
  } greater;

} // namespace dynd::nd
} // namespace dynd