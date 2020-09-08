#if !defined(DYND_ABI_TYPES_DENSE_H)
#define DYND_ABI_TYPES_DENSE_H

#include "dynd/abi/type.h"
#include "dynd/abi/type_constructor.h"
#include "dynd/abi/types/prefix.h"

#if defined(__cplusplus)
extern "C" {
#endif // defined(__cplusplus)

#define dynd_type_dense_typemeta DYND_TYPE(dense_typemeta)
typedef struct {
  dynd_type *parameter;
} dynd_type_dense_typemeta;

#define dynd_type_dense_arrmeta DYND_TYPE(dense_arrmeta)
typedef struct {
  dynd_size_t size;
  dynd_size_t stride;
} dynd_type_dense_arrmeta;

// The actual layout of a dense type returned from the
// dense type constructor.
#define dynd_type_dense_concrete DYND_TYPE(dense_concrete)
typedef struct {
  dynd_type prefix;
  dynd_type_dense_typemeta typemeta;
} dynd_type_dense_concrete;

#define dynd_type_dense DYND_TYPE(dense)

#if !defined(DYND_ABI_TYPES_DENSE_CPP)
extern DYND_ABI_EXPORT dynd_type_constructor dynd_type_dense;
#endif // !defined(DYND_ABI_TYPES_DENSE_CPP)

#if defined(__cplusplus)
}
#endif // defined(__cplusplus)

#endif // !defined(DYND_ABI_TYPES_DENSE_H)
