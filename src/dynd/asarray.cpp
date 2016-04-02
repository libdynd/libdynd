//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/asarray.hpp>
#include <dynd/types/substitute_shape.hpp>

using namespace std;
using namespace dynd;

nd::array nd::asarray(const nd::array &a, const ndt::type &tp)
{
  // TODO: This could attempt to do some more sophisticated views like
  //       "var * int" viewed as "Fixed * int". It's probably not reasonable to
  //       actually use nd::view here, because of nd::view's ability to view POD
  //       data as a different type.
  nd::array result;
  if (tp.match(a.get_type())) {
    // It already matches, just pass along the array
    result = a;
  }
  else if ((tp.get_flags() & type_flag_symbolic) == 0) {
    // The requested type is concrete, so can make an empty one easily
    result = nd::empty(tp);
    result.vals() = a;
  }
  else {
    // The requested type is symbolic, so need to fill in missing bits from the
    // input to make it concrete
    intptr_t ndim = a.get_ndim();
    dimvector shape(ndim);
    a.get_shape(shape.get());
    result = nd::empty(ndt::substitute_shape(tp, ndim, shape.get()));
    result.vals() = a;
  }

  return result;
}
