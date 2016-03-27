//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/callable.hpp>

namespace dynd {
namespace nd {

  /**
   * If possible returns an array which is a view of 'arr'.
   * When the types match exactly, returns 'arr' itself.
   *
   * Raises a type_error if the view is not possible.
   *
   * \param arr  The array to view as another type.
   * \param tp  The type to view the array as.
   */
  DYND_API array old_view(const array &arr, const ndt::type &tp);

  /**
   * Convenience form of nd::view that takes a string
   * literal instead of an ndt::type.
   */
  template <int N>
  inline array old_view(const array &arr, const char(&tp)[N])
  {
    return old_view(arr, ndt::type(tp, tp + N - 1));
  }

  extern DYND_API callable view;

} // namespace dynd::nd
} // namespace dynd
