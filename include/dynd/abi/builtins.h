#if !defined(DYND_ABI_BUILTINS_H)
#define DYND_ABI_BUILTINS_H

#include "dynd/abi/type.h"
#include "dynd/abi/version.h"

// TODO: Expand the DYND_TYPE_IMPL macro into something more generally useful?
#define DYND_TYPE_IMPL(prefix, name) DYND_ABI(prefix##name)
#define DYND_TYPE(name) DYND_TYPE_IMPL(type_, name)

#if defined(__cplusplus)
extern "C" {
#endif // defined(__cplusplus)

#define dynd_type_primitive_typemeta DYND_TYPE(primitive_typemeta)
typedef struct {
  dynd_size_t size;
  dynd_size_t alignment;
} dynd_type_primitive_typemeta;

#define dynd_type_primitive DYND_TYPE(primitive)
typedef struct {
  dynd_type prefix;
  dynd_type_primitive_typemeta typemeta;
} dynd_type_primitive;

#define dynd_type_float16 DYND_TYPE(float16)
extern DYND_ABI_EXPORT dynd_type_primitive dynd_types_float16;
#define dynd_type_float32 DYND_TYPE(float32)
extern DYND_ABI_EXPORT dynd_type_primitive dynd_types_float32;
#define dynd_type_float64 DYND_TYPE(float64)
extern DYND_ABI_EXPORT dynd_type_primitive dynd_types_float64;

#define dynd_type_uint8 DYND_TYPE(uint8)
extern DYND_ABI_EXPORT dynd_type_primitive dynd_types_uint8;
#define dynd_type_uint16 DYND_TYPE(uint16)
extern DYND_ABI_EXPORT dynd_type_primitive dynd_types_uint16;
#define dynd_type_uint32 DYND_TYPE(uint32)
extern DYND_ABI_EXPORT dynd_type_primitive dynd_types_uint32;
#define dynd_type_uint64 DYND_TYPE(uint64)
extern DYND_ABI_EXPORT dynd_type_primitive dynd_types_uint64;
#define dynd_type_int8 DYND_TYPE(int8)
extern DYND_ABI_EXPORT dynd_type_primitive dynd_types_int8;
#define dynd_type_int16 DYND_TYPE(int16)
extern DYND_ABI_EXPORT dynd_type_primitive dynd_types_int16;
#define dynd_type_int32 DYND_TYPE(int32)
extern DYND_ABI_EXPORT dynd_type_primitive dynd_types_int32;
#define dynd_type_int64 DYND_TYPE(int64)
extern DYND_ABI_EXPORT dynd_type_primitive dynd_types_int64;

#define dynd_type_size_t DYND_TYPE(size_t)
extern DYND_ABI_EXPORT dynd_type_primitive dynd_types_size_t;

#if defined(__cplusplus)
}
#endif // defined(__cplusplus)

#endif // !defined(DYND_ABI__H)
