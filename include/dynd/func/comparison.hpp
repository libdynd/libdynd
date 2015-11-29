//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/func/callable.hpp>
#include <dynd/func/call.hpp>
#include <dynd/func/elwise.hpp>
#include <dynd/func/multidispatch.hpp>
#include <dynd/kernels/compare_kernels.hpp>
#include <dynd/types/option_type.hpp>

namespace dynd {
namespace nd {

  template <typename FuncType, template <type_id_t...> class KernelType>
  struct comparison_operator : declfunc<FuncType> {
    static decltype(auto) make_children()
    {
      typedef type_id_sequence<bool_type_id, int8_type_id, int16_type_id, int32_type_id, int64_type_id, uint8_type_id,
                               uint16_type_id, uint32_type_id, uint64_type_id, float32_type_id,
                               float64_type_id> numeric_type_ids;

      auto children = callable::make_all<KernelType, numeric_type_ids, numeric_type_ids>();

      callable self = functional::call<FuncType>(ndt::type("(Any, Any) -> Any"));

      for (type_id_t i0 : i2a<numeric_type_ids>()) {
        for (type_id_t i1 : i2a<dim_type_ids>()) {
          const ndt::type child_tp = ndt::callable_type::make(ndt::type("Any"), {ndt::type(i0), ndt::type(i1)});
          children[{{i0, i1}}] = functional::elwise(child_tp, self);
        }
      }

      for (type_id_t i : i2a<numeric_type_ids>()) {
        children[{{option_type_id, i}}] = callable::make<option_comparison_kernel<FuncType, true, false>>();
        children[{{i, option_type_id}}] = callable::make<option_comparison_kernel<FuncType, false, true>>();
      }
      children[{{option_type_id, option_type_id}}] = callable::make<option_comparison_kernel<FuncType, true, true>>();

      for (type_id_t dim_tp_id : i2a<dim_type_ids>()) {
        children[{{dim_tp_id, option_type_id}}] = functional::elwise(self);
        children[{{option_type_id, dim_tp_id}}] = functional::elwise(self);
      }

      for (type_id_t i0 : i2a<dim_type_ids>()) {
        typedef join<numeric_type_ids, dim_type_ids>::type type_ids;
        for (type_id_t i1 : i2a<type_ids>()) {
          const ndt::type child_tp = ndt::callable_type::make(ndt::type("Any"), {ndt::type(i0), ndt::type(i1)});
          children[{{i0, i1}}] = functional::elwise(child_tp, self);
        }
      }

      children[{{string_type_id, string_type_id}}] = callable::make<KernelType<string_type_id, string_type_id>>();

      return children;
    }

    static callable make()
    {
      auto children = FuncType::make_children();
      return functional::multidispatch(
          ndt::type("(Any, Any) -> Any"), [children](const ndt::type &DYND_UNUSED(dst_tp), intptr_t DYND_UNUSED(nsrc),
                                                     const ndt::type *src_tp) mutable -> callable & {
            callable &child = children[{{src_tp[0].get_type_id(), src_tp[1].get_type_id()}}];
            if (child.is_null()) {
              throw std::runtime_error("no child found");
            }

            return child;
          });
    }
  };

  extern DYND_API struct less : comparison_operator<less, less_kernel> {
  } less;

  extern DYND_API struct less_equal : comparison_operator<less_equal, less_equal_kernel> {
  } less_equal;

  extern DYND_API struct equal : comparison_operator<equal, equal_kernel> {
    static std::map<std::array<type_id_t, 2>, callable> make_children()
    {
      std::map<std::array<type_id_t, 2>, callable> children = comparison_operator::make_children();
      children[{{complex_float32_type_id, complex_float32_type_id}}] =
          callable::make<equal_kernel<complex_float32_type_id, complex_float32_type_id>>(0);
      children[{{complex_float64_type_id, complex_float64_type_id}}] =
          callable::make<equal_kernel<complex_float64_type_id, complex_float64_type_id>>(0);
      children[{{tuple_type_id, tuple_type_id}}] = callable::make<equal_kernel<tuple_type_id, tuple_type_id>>(0);
      children[{{struct_type_id, struct_type_id}}] = callable::make<equal_kernel<tuple_type_id, tuple_type_id>>(0);
      children[{{type_type_id, type_type_id}}] = callable::make<equal_kernel<type_type_id, type_type_id>>(0);

      return children;
    }
  } equal;

  extern DYND_API struct not_equal : comparison_operator<not_equal, not_equal_kernel> {
    static std::map<std::array<type_id_t, 2>, callable> make_children()
    {
      std::map<std::array<type_id_t, 2>, callable> children = comparison_operator::make_children();
      children[{{complex_float32_type_id, complex_float32_type_id}}] =
          callable::make<not_equal_kernel<complex_float32_type_id, complex_float32_type_id>>(0);
      children[{{complex_float64_type_id, complex_float64_type_id}}] =
          callable::make<not_equal_kernel<complex_float64_type_id, complex_float64_type_id>>(0);
      children[{{tuple_type_id, tuple_type_id}}] = callable::make<not_equal_kernel<tuple_type_id, tuple_type_id>>(0);
      children[{{struct_type_id, struct_type_id}}] = callable::make<not_equal_kernel<tuple_type_id, tuple_type_id>>(0);
      children[{{type_type_id, type_type_id}}] = callable::make<not_equal_kernel<type_type_id, type_type_id>>(0);

      return children;
    }
  } not_equal;

  extern DYND_API struct greater_equal : comparison_operator<greater_equal, greater_equal_kernel> {
  } greater_equal;

  extern DYND_API struct greater : comparison_operator<greater, greater_kernel> {
  } greater;

  extern DYND_API struct total_order : declfunc<total_order> {
    static DYND_API callable make();
  } total_order;

} // namespace dynd::nd
} // namespace dynd
