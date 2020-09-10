#include <array>
#include <iostream>
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

// Case-specific arrmeta used below.
// The portions of this struct are type-specific ABIs that define the needed
// metadata for a container of the given type.
// Although this isn't strictly necessary, it helps to lower the number of
// reinterpret cast operations later on.
// TODO: There should probably just be ways to compute arrmeta sizes given
// the desired type. This should be added as an entry in the type vtable.
struct full_arrmeta {
  // arrmeta from the outer dense portion of the type.
  dynd_size_t num_nodes;
  dynd_size_t node_stride;
  // arrmeta from the tuple type stored for each row.
  dynd_size_t row_tuple_num_entries; // In this case we know this is just 2.
  // The offset for the node_data member of this tuple is implicitly 0 and is not stored.
  dynd_size_t row_topology_offset;
  // arrmeta from the node data.
  dynd_size_t node_data_num_entries;
  dynd_size_t float1_offset;
  dynd_size_t float2_offset;
  // arrmeta from the sparse dimension type used to encode the edges for the given node.
  // TODO: The empty edge data case is relatively common. How can we avoid storing the
  // extra offset and stride?
  dynd_size_t logical_size;
  dynd_size_t indptr_stride;
  dynd_size_t indices_offset;
  dynd_size_t indices_stride;
  dynd_size_t data_offset;
  dynd_size_t data_stride;
  // The arrmeta for the empty tuple given as edge data.
  // Will always be 0.
  dynd_size_t empty_tuple_num_entries;
};

// Also for convenience.
// In the demo we slice away node_data entries that aren't used
// in a given kernel call. This is the arrmeta layout for the resulting
// view into the original data buffer. Again, these are type-specific ABIs.
struct sliced_arrmeta {
  dynd_size_t num_nodes;
  dynd_size_t node_stride;
  dynd_size_t row_tuple_num_entries;
  dynd_size_t row_topology_offset;
  dynd_size_t logical_size;
  dynd_size_t indptr_stride;
  dynd_size_t indices_offset;
  dynd_size_t indices_stride;
  dynd_size_t data_offset;
  dynd_size_t data_stride;
  dynd_size_t empty_tuple_num_entries;
};

// Slice a CSR graph to select a specific entry of the node data.
dynd_array *select_node_data_entry(dynd_array *array) noexcept; // TODO

// TODO: Actually, taking the address of the builtin types/type constructors
// is a bit cumbersome. It may be better to just make the pointers to them
// be the user-exposed interface.

int main() {
  // node_data = tuple(size_t, float64, float64)
  dynd_type *node_data = invoke_constructor(&dynd_type_tuple, &dynd_type_size_t, &dynd_type_float64, &dynd_type_float64);
  // edge_data = tuple()
  // TODO: Should there be a primitive type for this singleton?
  dynd_type *edge_data = invoke_constructor(&dynd_type_tuple);
  // row_topology = sparse(edge_data)
  dynd_type *row_topology = invoke_constructor(&dynd_type_sparse, edge_data);
  // row_data = tuple(node_data, row_topology)
  dynd_type *row_data = invoke_constructor(&dynd_type_tuple, node_data, row_topology);
  // csr_graph = dense(row_data)
  // In other words
  // csr_graph = dense(tuple(node_data, sparse(edge_data)))
  // Note: in-line vs out-of-line edge data, AOS vs SOA layouts
  // for the edge data and node data, and interleaving the node data
  // with the indptr entries for each node vs storing it out-of-line
  // are all things that can be handled by configuring the metadata
  // in the corresponding dynd_array handle in different ways.
  dynd_type *csr_graph = invoke_constructor(&dynd_type_dense, row_data);
  // Now use csr_graph to prevent the compiler from griping about an
  // unused variable.
  std::cout << csr_graph << std::endl;
}
