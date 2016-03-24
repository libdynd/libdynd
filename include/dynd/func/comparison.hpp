//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/callable.hpp>
#include <dynd/functional.hpp>
#include <dynd/kernels/compare_kernels.hpp>
#include <dynd/types/option_type.hpp>
#include <dynd/callables/comparison_dispatch_callable.hpp>
#include <dynd/callables/less_callable.hpp>
#include <dynd/callables/less_equal_callable.hpp>
#include <dynd/callables/equal_callable.hpp>
#include <dynd/callables/not_equal_callable.hpp>
#include <dynd/callables/greater_equal_callable.hpp>
#include <dynd/callables/greater_callable.hpp>

namespace dynd {
namespace nd {

  template <typename FuncType, template <type_id_t...> class KernelType>
  struct comparison_operator : declfunc<FuncType> {
    static decltype(auto) make_children()
    {
      typedef type_id_sequence<bool_id, int8_id, int16_id, int32_id, int64_id, uint8_id, uint16_id, uint32_id,
                               uint64_id, float32_id, float64_id> numeric_ids;

      auto dispatcher = callable::new_make_all<KernelType, numeric_ids, numeric_ids>();

      callable self = functional::call<FuncType>(ndt::type("(Any, Any) -> Any"));

      for (type_id_t i0 : i2a<numeric_ids>()) {
        for (type_id_t i1 : i2a<dim_ids>()) {
          const ndt::type child_tp = ndt::callable_type::make(ndt::type("Any"), {ndt::type(i0), ndt::type(i1)});
          dispatcher.insert({{i0, i1}, functional::elwise(child_tp, self)});
        }
      }

      for (type_id_t i : i2a<numeric_ids>()) {
        dispatcher.insert({{option_id, i}, functional::forward_na<0>(self)});
        dispatcher.insert({{i, option_id}, functional::forward_na<1>(self)});
      }
      dispatcher.insert({{option_id, option_id}, make_callable<option_comparison_callable<FuncType, true, true>>()});

      for (type_id_t dim_tp_id : i2a<dim_ids>()) {
        dispatcher.insert({{dim_tp_id, option_id}, functional::elwise(self)});
        dispatcher.insert({{option_id, dim_tp_id}, functional::elwise(self)});
      }

      for (type_id_t i0 : i2a<dim_ids>()) {
        typedef join<numeric_ids, dim_ids>::type type_ids;
        for (type_id_t i1 : i2a<type_ids>()) {
          const ndt::type child_tp = ndt::callable_type::make(ndt::type("Any"), {ndt::type(i0), ndt::type(i1)});
          dispatcher.insert({{i0, i1}, functional::elwise(child_tp, self)});
        }
      }

      dispatcher.insert({{string_id, string_id}, make_callable<KernelType<string_id, string_id>>()});

      return dispatcher;
    }

    static callable make()
    {
      auto dispatcher = FuncType::make_children();
      return make_callable<comparison_dispatch_callable>(ndt::type("(Any, Any) -> Any"), dispatcher);
    }
  };

  extern DYND_API struct DYND_API less : comparison_operator<less, less_callable> {
    static callable &get();
  } less;

  extern DYND_API struct DYND_API less_equal : comparison_operator<less_equal, less_equal_callable> {
    static callable &get();
  } less_equal;

  extern DYND_API struct DYND_API equal : comparison_operator<equal, equal_callable> {
    static dispatcher<callable> make_children()
    {
      auto dispatcher = comparison_operator::make_children();
      dispatcher.insert({{complex_float32_id, complex_float32_id},
                         make_callable<equal_callable<complex_float32_id, complex_float32_id>>()});
      dispatcher.insert({{complex_float64_id, complex_float64_id},
                         make_callable<equal_callable<complex_float64_id, complex_float64_id>>()});
      dispatcher.insert({{tuple_id, tuple_id}, make_callable<equal_callable<tuple_id, tuple_id>>()});
      dispatcher.insert({{struct_id, struct_id}, make_callable<equal_callable<tuple_id, tuple_id>>()});
      dispatcher.insert({{type_id, type_id}, make_callable<equal_callable<type_id, type_id>>()});
      dispatcher.insert({{bytes_id, bytes_id}, make_callable<equal_callable<bytes_id, bytes_id>>()});

      return dispatcher;
    }
    static callable &get();
  } equal;

  extern DYND_API struct DYND_API not_equal : comparison_operator<not_equal, not_equal_callable> {
    static dispatcher<callable> make_children()
    {
      auto dispatcher = comparison_operator::make_children();
      dispatcher.insert({{complex_float32_id, complex_float32_id},
                         make_callable<not_equal_callable<complex_float32_id, complex_float32_id>>()});
      dispatcher.insert({{complex_float64_id, complex_float64_id},
                         make_callable<not_equal_callable<complex_float64_id, complex_float64_id>>()});
      dispatcher.insert({{tuple_id, tuple_id}, make_callable<not_equal_callable<tuple_id, tuple_id>>()});
      dispatcher.insert({{struct_id, struct_id}, make_callable<not_equal_callable<tuple_id, tuple_id>>()});
      dispatcher.insert({{type_id, type_id}, make_callable<not_equal_callable<type_id, type_id>>()});
      dispatcher.insert({{bytes_id, bytes_id}, make_callable<not_equal_callable<bytes_id, bytes_id>>()});

      return dispatcher;
    }
    static callable &get();
  } not_equal;

  extern DYND_API struct DYND_API greater_equal : comparison_operator<greater_equal, greater_equal_callable> {
    static callable &get();
  } greater_equal;

  extern DYND_API struct DYND_API greater : comparison_operator<greater, greater_callable> {
    static callable &get();
  } greater;

  extern DYND_API struct DYND_API total_order : declfunc<total_order> {
    static callable make();
    static callable &get();
  } total_order;

} // namespace dynd::nd
} // namespace dynd
