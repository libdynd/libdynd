//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/array.hpp>

namespace dynd {
namespace nd {
  // TODO: This might be a good place to put a customization point
  // (http://ericniebler.com/2014/10/21/customization-point-design-in-c11-and-beyond/)
  // for an asarray(T&& a) variant which views C++ objects as nd::array by
  // viewing their data where this makes sense. The
  // nd::asarray(T&& a, const ndt::type &tp) would then call
  // nd::asarray(nd::asarray(a), tp) to get to the view an array as a
  // particular type version.

  /**
   * This function passes along `a` or produces a view of `a` compatible
   * with the type `tp` if that is possible, otherwise attempts to construct a
   * new array of the provided `tp` initialized to the values of `a`. If this
   * is not possible, it raises an exception.
   *
   * \param a  The input array.
   * \param tp  The type the result should be compatible with. This may be
   *            concrete, in which case the output will have exactly the type,
   *            or symbolic, in which case the output will have a type which
   *            matches successfully against `tp`.
   */
  DYND_API nd::array asarray(const nd::array &a, const ndt::type &tp);

} // namespace nd
} // namespace dynd
