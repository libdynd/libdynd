#if !defined(DYND_ABI_GENERIC_FUNCTION_POINTER_H)
#define DYND_ABI_GENERIC_FUNCTION_POINTER_H

#if defined(__cplusplus) && __cplusplus >= 201103L && __cplusplus < 201703L
// Needed for trick to get noexcept into function pointer typedef
// in C++11 and C++14.
#include <type_traits>
#include <utility>
#endif // defined(__cplusplus) && __cplusplus >= 201103L && __cplusplus < 201703L

#include "dynd/abi/noexcept.h"

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

// Defining a function pointer typedef to be noexcept
// in language standards that permit that
#if defined(__cplusplus) && __cplusplus >= 201703L
#define DYND_ABI_NOEXCEPT_FUNC(name, ret_type, ...) typedef ret_type (*name)(__VA_ARGS__) dynd_noexcept;
#else // defined(__cplusplus) && __cplusplus >= 201703L
// noexcept isn't officially a part of a function pointer type
// in C++11 and C++14. Though there are some tricks to get
// sort-of similar behavior, they are unreliable with current
// compilers, so just do a plain C-style function pointer typedef.
// No way to include throw() in the function pointer type
// in C++98 either, so do the same thing in that case.
#define DYND_ABI_NOEXCEPT_FUNC(name, ret_type, ...) typedef ret_type (*name)(__VA_ARGS__);
#endif

#endif // !defined(DYND_ABI_GENERIC_FUNCTION_POINTER_H)
