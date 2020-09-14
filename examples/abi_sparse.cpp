#include <array>
#include <cassert>
#include <cstdint>
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
struct sliced_array {
  dynd_array header;
  sliced_arrmeta arrmeta;
};

// Slice a CSR graph to select a specific entry of the node data.
// TODO: Types need a vtable entry that enables getting the arrmeta
// associated with the given type without dealing with the hassle of
// tracking through all the arrmeta in the array manually.
// It's not obvious what that interface needs to look like yet.
// In other words, I shouldn't necessarily have to know about full_arrmeta
// to build the corresponding sliced_arrmeta.
dynd_array *select_node_data_entry(dynd_array *array) noexcept {
  // This works assuming that the array has the type csr_graph
  // defined in the main function.
  full_arrmeta &full = *reinterpret_cast<full_arrmeta*>(dynd_array_metadata(array));
  dynd_array *result = reinterpret_cast<dynd_array*>(dynd_malloc_buffer(sizeof(sliced_array)));
  sliced_arrmeta &sliced = *reinterpret_cast<sliced_arrmeta*>(dynd_array_metadata(result));
  // Since the first tuple element in the per-edge data is the only one we want,
  // and the first tuple element is defined to have zero offset, we can just
  // copy the entries we need from the previous array's arrmeta without modification.
  sliced.num_nodes = full.num_nodes;
  sliced.node_stride = full.node_stride;
  sliced.row_tuple_num_entries = full.row_tuple_num_entries;
  sliced.row_topology_offset = full.row_topology_offset;
  sliced.logical_size = full.logical_size;
  sliced.indptr_stride = full.indptr_stride;
  sliced.indices_offset = full.indices_offset;
  sliced.indices_stride = full.indices_stride;
  sliced.data_offset = full.data_offset;
  sliced.data_stride = full.data_stride;
  sliced.empty_tuple_num_entries = full.empty_tuple_num_entries;
  // Set up the members of the actual dynd_array struct
  result->header.base_array = array;
  result->header.base = array->header.base;
  result->header.access = array->header.access;
  // Now set up the type of the new array
  // This mimics the logic in main for building the csr matrix type for the full array.
  dynd_type *node_data = &dynd_type_size_t;
  dynd_type *edge_data = invoke_constructor(&dynd_type_tuple);
  dynd_type *row_topology = invoke_constructor(&dynd_type_sparse, edge_data);
  dynd_type *row_data = invoke_constructor(&dynd_type_tuple, node_data, row_topology);
  dynd_type *sliced_csr_graph = invoke_constructor(&dynd_type_dense, row_data);
  result->header.type = sliced_csr_graph;
  return result;
}

// Wraparound arithmetic is part of what
// allows storing arbitrary tuples as offsets
// from a single base pointer. Overflowing a
// pointer is undefined behavior though, so
// we have to do this ugly casting to get
// the wraparound behavior that is allowed
// for unsigned types.
void *wraparound_pointer_add(void *ptr, dynd_size_t offset) noexcept {
  return reinterpret_cast<void*>(reinterpret_cast<dynd_size_t>(ptr) + offset);
}

// Type-specific indexing function for the sliced array.
// This operation will ultimately be doable via a
// a multiple dispatch system, but this serves as a
// stopgap until then.
dynd_size_t &node_data(dynd_array *array, dynd_size_t id) noexcept {
  static_assert(sizeof(dynd_size_t) == sizeof(std::intptr_t));
  sliced_arrmeta &arrmeta = *reinterpret_cast<sliced_arrmeta*>(dynd_array_metadata(array));
  void *base = array->header.base;
  return *reinterpret_cast<dynd_size_t*>(wraparound_pointer_add(base, id * arrmeta.node_stride));
}

// Type-specific access to the first entry of
// the indptr array for a given node.
// Note: the last entry can be set by using
// an index one higher than the highest node id.
// This is another operation that will be
// better handled once there's a working multiple dispatch system.
dynd_size_t &first_indptr(dynd_array *array, dynd_size_t id) noexcept {
  sliced_arrmeta &arrmeta = *reinterpret_cast<sliced_arrmeta*>(dynd_array_metadata(array));
  void *base = array->header.base;
  return *reinterpret_cast<dynd_size_t*>(wraparound_pointer_add(base, arrmeta.row_topology_offset + id * arrmeta.indptr_stride));
}

// Indexing routine to access the indices array
dynd_size_t &indices(dynd_array *array, dynd_size_t idx) noexcept {
  sliced_arrmeta &arrmeta = *reinterpret_cast<sliced_arrmeta*>(dynd_array_metadata(array));
  void *base = array->header.base;
  return *reinterpret_cast<dynd_size_t*>(wraparound_pointer_add(base, arrmeta.row_topology_offset + arrmeta.indices_offset + idx * arrmeta.indices_stride));
}

struct neighbor_range{
  dynd_size_t *begin;
  dynd_size_t *end;
  dynd_size_t stride;
};

// Type-specific neighbor range function for the sliced array.
// The stride between nodes could technically be inferred from the arrmeta
// in the outer function, but including that in the return value
// here seemed like it'd simplify things overall.
neighbor_range neighbors(dynd_array *array, dynd_size_t id) noexcept {
  sliced_arrmeta &arrmeta = *reinterpret_cast<sliced_arrmeta*>(dynd_array_metadata(array));
  void *base = array->header.base;
  void *node_base = wraparound_pointer_add(base, id * arrmeta.node_stride);
  void *indptr_base = wraparound_pointer_add(node_base, arrmeta.row_topology_offset);
  dynd_size_t start_index = *reinterpret_cast<dynd_size_t*>(indptr_base);
  dynd_size_t end_index = *reinterpret_cast<dynd_size_t*>(wraparound_pointer_add(indptr_base, arrmeta.indptr_stride));
  void *indices_base = wraparound_pointer_add(node_base, arrmeta.indices_offset);
  dynd_size_t *begin = reinterpret_cast<dynd_size_t*>(wraparound_pointer_add(indices_base, start_index * arrmeta.indices_stride));
  dynd_size_t *end = reinterpret_cast<dynd_size_t*>(wraparound_pointer_add(indices_base, end_index * arrmeta.indices_stride));
  dynd_size_t typed_stride = arrmeta.indices_stride / sizeof(dynd_size_t);
  assert(!(arrmeta.indices_stride % alignof(dynd_size_t)));
  return {begin, end, typed_stride};
}

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
