#if !defined(DYND_ABI_TYPES_TUPLE_H)
#define DYND_ABI_TYPES_TUPLE_H

#include "dynd/abi/type.h"
#include "dynd/abi/type_constructor.h"
#include "dynd/abi/types/prefix.h"

#if defined(__cplusplus)
extern "C" {
#endif // defined(__cplusplus)

#define dynd_type_tuple_typemeta_header DYND_TYPE(tuple_typemeta_header)
typedef struct {
  dynd_size_t num_entries;
} dynd_type_tuple_typemeta_header;

#define dynd_type_tuple_arrmeta_header DYND_TYPE(tuple_arrmeta_header)
typedef struct {
  dynd_size_t num_entries;
} dynd_type_tuple_arrmeta_header;

#define dynd_type_tupl_arrmeta_entry DYND_TYPE(tuple_arrmeta_entry)
typedef struct {
  dynd_size_t offset;
  dynd_size_t stride;
} dynd_type_tuple_arrmeta_entry;

// Unlike in the dense case, the typemeta of the generated
// tuple type depends on the number of input parameters.
// This makes it impossible to fully write out the generic
// tuple typemeta layout for every possible tuple here, but
// in all cases this much of the header is consistent.
#define dynd_type_tuple_concrete_header DYND_TYPE(tuple_concrete_header)
typedef struct {
  dynd_type prefix;
  dynd_type_tuple_typemeta_header typemeta_header;
} dynd_type_tuple_concrete_header;

#define dynd_type_tuple DYND_TYPE(tuple)
#if !defined(DYND_ABI_TYPES_TUPLE_CPP)
extern DYND_ABI_EXPORT dynd_type_constructor dynd_type_tuple;
#endif // !defined(DYND_ABI_TYPES_TUPLE_CPP)

#if defined(__cplusplus)
}
#endif // defined(__cplusplus)

#endif // !defined(DYND_ABI_TYPES_TUPLE_H)
