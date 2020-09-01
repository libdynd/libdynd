#if !defined(DYND_ABI_GENERIC_FUNCTION_POINTER_H)
#define DYND_ABI_GENERIC_FUNCTION_POINTER_H

#if defined(__cplusplus) && __cplusplus >= 201103L && __cplusplus < 201703L
// Needed for trick to get noexcept into function pointer typedef
// in C++11 and C++14.
#include <type_traits>
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
#elif defined(__cplusplus) && __cplusplus >= 201103L
// C++ equivalent of C typedef with the noexcept specifier added.
// Note: this isn't possible with a plain typedef.
// Use the technique from: https://stackoverflow.com/a/53674998
// Noexcept is a full part of the type system in c++17 and later,
// but this trickery is needed to get the right behavior
// with earlier versions that still support noexcept.
// Note: This causes compiler bugs when it's default-initialized
// as a struct member, so we can't use this trick for now.
//#define DYND_ABI_NOEXCEPT_FUNC(name, ret_type, ...) using name = decltype(std::declval<ret_type(*)(__VA_ARGS__) dynd_noexcept>());
#define DYND_ABI_NOEXCEPT_FUNC(name, ret_type, ...) typedef ret_type (*name)(__VA_ARGS__);
#else
// No way to include throw() in the function pointer type,
// so just use the equivalent c typedef.
#define DYND_ABI_NOEXCEPT_FUNC(name, ret_type, ...) typedef ret_type (*name)(__VA_ARGS__);
#endif

#endif // !defined(DYND_ABI_GENERIC_FUNCTION_POINTER_H)
