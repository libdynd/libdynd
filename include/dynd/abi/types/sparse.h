#if !defined(DYND_ABI_TYPES_SPARSE_H)
#define DYND_ABI_TYPES_SPARSE_H

#include "dynd/abi/type.h"
#include "dynd/abi/type_constructor.h"
#include "dynd/abi/types/prefix.h"

#if defined(__cplusplus)
extern "C" {
#endif // defined(__cplusplus)

// For the time being, just assume indices are size_t.
// This makes it so that the only needed parameter to the
// type constructor is the item type for the sparse dimension.
#define dynd_type_sparse_typemeta DYND_TYPE(sparse_typemeta)
typedef struct {
  dynd_type *parameter;
} dynd_type_sparse_typemeta;

// Here's how indexing works with a sparse dimension:
// - the logical size has to be stored explicitly.
// - indptr is conceptually a dense array with only two entries.
// - indptr for a csr matrix can be stored in the usual way by
//   having a stride of 1 * sizeof(size_t) as the stride between
//   the 2D entries that correspond to the start/end indices for
//   each row.
// - indptr stores offsets into the indices dense array which may
//   have many entries. Its size isn't stored explicitly since
//   indptr array already provides the bounds we need.
// - For each entry in indices, there's also an entry in data.
//   the same reasoning applies for not storing the size of the data.
// - The convention we've chosen here is that indptr is at offset 0,
//   so there's no need to store the offset off of the base pointer
//   for that array. The indices and data arrays still need offsets.
#define dynd_type_sparse_arrmeta DYND_TYPE(sparse_arrmeta)
typedef struct {
  dynd_size_t logical_size;
  dynd_size_t indptr_stride;
  dynd_size_t indices_offset;
  dynd_size_t indices_stride;
  dynd_size_t data_offset;
  dynd_size_t data_stride;
} dynd_type_sparse_arrmeta;

#define dynd_type_sparse_concrete DYND_TYPE(sparse_concrete)
typedef struct {
  dynd_type prefix;
  dynd_type_sparse_typemeta typemeta;
} dynd_type_sparse_concrete;

#define dynd_type_sparse DYND_TYPE(sparse)

#if !defined(DYND_ABI_TYPES_SPARSE_CPP)
extern DYND_ABI_EXPORT dynd_type_constructor dynd_type_sparse;
#endif // !defined(DYND_ABI_TYPES_SPARSE_CPP)

#if defined(__cplusplus)
}
#endif // defined(__cplusplus)

#endif // !defined(DYND_ABI_TYPES_SPARSE_H)
