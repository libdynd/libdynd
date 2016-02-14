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

  template <typename SelfType, template <type_id_t...> class KernelType>
  struct comparison_operator : declfunc<SelfType> {
    static std::map<std::array<type_id_t, 2>, callable> make_children();
    static callable make();
  };

  extern DYND_API struct DYND_API less : comparison_operator<less, less_kernel> {
  } less;

  extern DYND_API struct DYND_API less_equal : comparison_operator<less_equal, less_equal_kernel> {
  } less_equal;

  extern DYND_API struct DYND_API equal : comparison_operator<equal, equal_kernel> {
    static std::map<std::array<type_id_t, 2>, callable> make_children()
    {
      std::map<std::array<type_id_t, 2>, callable> children = comparison_operator::make_children();
      children[{{complex_float32_id, complex_float32_id}}] =
          callable::make<equal_kernel<complex_float32_id, complex_float32_id>>(0);
      children[{{complex_float64_id, complex_float64_id}}] =
          callable::make<equal_kernel<complex_float64_id, complex_float64_id>>(0);
      children[{{tuple_id, tuple_id}}] = callable::make<equal_kernel<tuple_id, tuple_id>>(0);
      children[{{struct_id, struct_id}}] = callable::make<equal_kernel<tuple_id, tuple_id>>(0);
      children[{{type_id, type_id}}] = callable::make<equal_kernel<type_id, type_id>>(0);

      return children;
    }
  } equal;

  extern DYND_API struct DYND_API not_equal : comparison_operator<not_equal, not_equal_kernel> {
    static std::map<std::array<type_id_t, 2>, callable> make_children()
    {
      std::map<std::array<type_id_t, 2>, callable> children = comparison_operator::make_children();
      children[{{complex_float32_id, complex_float32_id}}] =
          callable::make<not_equal_kernel<complex_float32_id, complex_float32_id>>(0);
      children[{{complex_float64_id, complex_float64_id}}] =
          callable::make<not_equal_kernel<complex_float64_id, complex_float64_id>>(0);
      children[{{tuple_id, tuple_id}}] = callable::make<not_equal_kernel<tuple_id, tuple_id>>(0);
      children[{{struct_id, struct_id}}] = callable::make<not_equal_kernel<tuple_id, tuple_id>>(0);
      children[{{type_id, type_id}}] = callable::make<not_equal_kernel<type_id, type_id>>(0);

      return children;
    }
  } not_equal;

  extern DYND_API struct DYND_API greater_equal : comparison_operator<greater_equal, greater_equal_kernel> {
  } greater_equal;

  extern DYND_API struct DYND_API greater : comparison_operator<greater, greater_kernel> {
  } greater;

  extern DYND_API struct DYND_API total_order : declfunc<total_order> {
    static callable make();
  } total_order;

} // namespace dynd::nd
} // namespace dynd
