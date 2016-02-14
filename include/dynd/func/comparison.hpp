//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/callable.hpp>

namespace dynd {
namespace nd {

  template <typename SelfType, template <type_id_t...> class KernelType>
  struct comparison_operator : declfunc<SelfType> {
    static std::map<std::array<type_id_t, 2>, callable> make_children();
    static callable make();
  };

  template <type_id_t Arg0ID, type_id_t Arg1ID>
  struct less_kernel;

  extern DYND_API struct DYND_API less : comparison_operator<less, less_kernel> {
  } less;

  template <type_id_t Arg0ID, type_id_t Arg1ID>
  struct less_equal_kernel;

  extern DYND_API struct DYND_API less_equal : comparison_operator<less_equal, less_equal_kernel> {
  } less_equal;

  template <type_id_t Arg0ID, type_id_t Arg1ID>
  struct equal_kernel;

  extern DYND_API struct DYND_API equal : comparison_operator<equal, equal_kernel> {
    static std::map<std::array<type_id_t, 2>, callable> make_children();
  } equal;

  template <type_id_t Arg0ID, type_id_t Arg1ID>
  struct not_equal_kernel;

  extern DYND_API struct DYND_API not_equal : comparison_operator<not_equal, not_equal_kernel> {
    static std::map<std::array<type_id_t, 2>, callable> make_children();
  } not_equal;

  template <type_id_t Arg0ID, type_id_t Arg1ID>
  struct greater_equal_kernel;

  extern DYND_API struct DYND_API greater_equal : comparison_operator<greater_equal, greater_equal_kernel> {
  } greater_equal;

  template <type_id_t Arg0ID, type_id_t Arg1ID>
  struct greater_kernel;

  extern DYND_API struct DYND_API greater : comparison_operator<greater, greater_kernel> {
  } greater;

  extern DYND_API struct DYND_API total_order : declfunc<total_order> {
    static callable make();
  } total_order;

} // namespace dynd::nd
} // namespace dynd
