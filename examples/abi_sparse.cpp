#include <array>
#include <type_traits>

#include <dynd/abi/array.h>
#include <dynd/abi/type.h>
#include <dynd/abi/type_constructor.h>
#include <dynd/abi/types/dense.h>
#include <dynd/abi/types/primitive.h>
#include <dynd/abi/types/sparse.h>
#include <dynd/abi/types/tuple.h>

// Utility functions that take shortcuts to provide things specific to this example.
// Many of these should be added to the main library, but the interface isn't settled yet.

// Invoke a type constructor with the given parameters.
// Returns an owned reference to the created type.
template <typename... type>
dynd_type * invoke_constructor(dynd_type_constructor *constructor, type*... types) noexcept {
  static_assert((std::is_same<type, dynd_type>::value && ... && true), "Constructors can currently only be passed types as parameters");
  std::array<dynd_type*, sizeof...(type)> parameters{types...};
  dynd_type_range type_range{parameters.begin(), parameters.end()};
  dynd_type_constructor_header *header = &constructor->header;
  return header->vtable->entries.make(header, type_range);
}

// Ad-hoc equality checking that works for the existing types.
// This probably needs a proper entry in the type vtable instead.
bool check_equal(dynd_type *left, dynd_type *right) noexcept {
  if (left->header.constructor == &dynd_type_make_primitive) return left == right;
  if (left->header.constructor == &dynd_type_dense || left->header.constructor == &dynd_type_sparse) {
    if (right->header.constructor != left->header.constructor) return false;
    return check_equal(*(left->header.vtable->entries.parameters(&left->header).begin), *(right->header.vtable->entries.parameters(&right->header).begin));
  }
  if (left->header.constructor == &dynd_type_tuple && right->header.constructor == &dynd_type_tuple) {
    dynd_type_range left_children = left->header.vtable->entries.parameters(&left->header);
    dynd_type_range right_children = right->header.vtable->entries.parameters(&right->header);
    while (left_children.begin < left_children.end && right_children.begin < right_children.end) {
      if (!check_equal(*left_children.begin, *right_children.begin)) return false;
      left_children.begin++;
      right_children.begin++;
    }
    if (left_children.begin < left_children.end || right_children.begin < right_children.end) return false;
    return true;
  }
  return false;
}

int main() {
  ;
}
