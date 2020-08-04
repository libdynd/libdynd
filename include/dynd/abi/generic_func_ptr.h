#if !defined(DYND_ABI_GENERIC_FUNCTION_POINTER_H)
#define DYND_ABI_GENERIC_FUNCTION_POINTER_H

// A function pointer type that's safe to cast
// other function pointers to/from.
// Since this struct type is never defined,
// it will always be an error if it is ever used
// without being cast to some other
// function pointer type.
// Don't ABI version this since it's really just
// more of a useful C idiom that we need.
struct dynd_abi_never_defined;
typedef void (*dynd_generic_func_ptr)(dynd_abi_never_defined);

#endif // !defined(DYND_ABI_GENERIC_FUNCTION_POINTER_H)
