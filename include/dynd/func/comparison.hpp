//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/callable.hpp>
#include <dynd/functional.hpp>
#include <dynd/kernels/compare_kernels.hpp>
#include <dynd/types/option_type.hpp>

namespace dynd {
namespace nd {

  template <typename FuncType, template <type_id_t...> class KernelType>
  struct comparison_operator : declfunc<FuncType> {
    static decltype(auto) make_children()
    {
      typedef type_id_sequence<bool_id, int8_id, int16_id, int32_id, int64_id, uint8_id, uint16_id, uint32_id,
                               uint64_id, float32_id, float64_id>
          numeric_ids;

      auto children = callable::make_all<KernelType, numeric_ids, numeric_ids>();

      callable self = functional::call<FuncType>(ndt::type("(Any, Any) -> Any"));

      for (type_id_t i0 : i2a<numeric_ids>()) {
        for (type_id_t i1 : i2a<dim_ids>()) {
          const ndt::type child_tp = ndt::callable_type::make(ndt::type("Any"), {ndt::type(i0), ndt::type(i1)});
          children[{{i0, i1}}] = functional::elwise(child_tp, self);
        }
      }

      for (type_id_t i : i2a<numeric_ids>()) {
        children[{{option_id, i}}] = functional::forward_na<0>(self);
        children[{{i, option_id}}] = functional::forward_na<1>(self);
      }
      children[{{option_id, option_id}}] = callable::make<option_comparison_kernel<FuncType, true, true>>();

      for (type_id_t dim_tp_id : i2a<dim_ids>()) {
        children[{{dim_tp_id, option_id}}] = functional::elwise(self);
        children[{{option_id, dim_tp_id}}] = functional::elwise(self);
      }

      for (type_id_t i0 : i2a<dim_ids>()) {
        typedef join<numeric_ids, dim_ids>::type type_ids;
        for (type_id_t i1 : i2a<type_ids>()) {
          const ndt::type child_tp = ndt::callable_type::make(ndt::type("Any"), {ndt::type(i0), ndt::type(i1)});
          children[{{i0, i1}}] = functional::elwise(child_tp, self);
        }
      }

      children[{{string_id, string_id}}] = callable::make<KernelType<string_id, string_id>>();

      return children;
    }

    static callable make()
    {
      auto children = FuncType::make_children();
      return functional::dispatch(ndt::type("(Any, Any) -> Any"),
                                  [children](const ndt::type &DYND_UNUSED(dst_tp), intptr_t DYND_UNUSED(nsrc),
                                             const ndt::type *src_tp) mutable -> callable & {
                                    callable &child = children[{{src_tp[0].get_id(), src_tp[1].get_id()}}];
                                    if (child.is_null()) {
                                      throw std::runtime_error("no child found");
                                    }

                                    return child;
                                  });
    }
  };

  extern DYND_API struct DYND_API less : comparison_operator<less, less_kernel> {
    static callable &get();
  } less;

  extern DYND_API struct DYND_API less_equal : comparison_operator<less_equal, less_equal_kernel> {
    static callable &get();
  } less_equal;

  extern DYND_API struct DYND_API equal : comparison_operator<equal, equal_kernel> {
    static std::map<std::array<type_id_t, 2>, callable> make_children()
    {
      std::map<std::array<type_id_t, 2>, callable> children = comparison_operator::make_children();
      children[{{complex_float32_id, complex_float32_id}}] =
          callable::make<equal_kernel<complex_float32_id, complex_float32_id>>();
      children[{{complex_float64_id, complex_float64_id}}] =
          callable::make<equal_kernel<complex_float64_id, complex_float64_id>>();
      children[{{tuple_id, tuple_id}}] = callable::make<equal_kernel<tuple_id, tuple_id>>();
      children[{{struct_id, struct_id}}] = callable::make<equal_kernel<tuple_id, tuple_id>>();
      children[{{type_id, type_id}}] = callable::make<equal_kernel<type_id, type_id>>();
      children[{{bytes_id, bytes_id}}] = callable::make<equal_kernel<bytes_id, bytes_id>>();

      return children;
    }
    static callable &get();
  } equal;

  extern DYND_API struct DYND_API not_equal : comparison_operator<not_equal, not_equal_kernel> {
    static std::map<std::array<type_id_t, 2>, callable> make_children()
    {
      std::map<std::array<type_id_t, 2>, callable> children = comparison_operator::make_children();
      children[{{complex_float32_id, complex_float32_id}}] =
          callable::make<not_equal_kernel<complex_float32_id, complex_float32_id>>();
      children[{{complex_float64_id, complex_float64_id}}] =
          callable::make<not_equal_kernel<complex_float64_id, complex_float64_id>>();
      children[{{tuple_id, tuple_id}}] = callable::make<not_equal_kernel<tuple_id, tuple_id>>();
      children[{{struct_id, struct_id}}] = callable::make<not_equal_kernel<tuple_id, tuple_id>>();
      children[{{type_id, type_id}}] = callable::make<not_equal_kernel<type_id, type_id>>();
      children[{{bytes_id, bytes_id}}] = callable::make<not_equal_kernel<bytes_id, bytes_id>>();

      return children;
    }
    static callable &get();
  } not_equal;

  extern DYND_API struct DYND_API greater_equal : comparison_operator<greater_equal, greater_equal_kernel> {
    static callable &get();
  } greater_equal;

  extern DYND_API struct DYND_API greater : comparison_operator<greater, greater_kernel> {
    static callable &get();
  } greater;

  extern DYND_API struct DYND_API total_order : declfunc<total_order> {
    static callable make();
    static callable &get();
  } total_order;

} // namespace dynd::nd
} // namespace dynd
