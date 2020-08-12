#if !defined(DYND_ABI_VISIBILITY_H)
#define DYND_ABI_VISIBILITY_H

// The new ABI isn't intended to be built as a static object,
// so that simplifies these macros significantly.

#if defined(_WIN32) || defined(__CYGWIN__)
// dllexport/dllimport are the right specifiers to use on Windows
// regardless of the compiler used.
#if defined(DYND_ABI_COMPILING)
#define DYND_ABI_EXPORT __declspec(dllexport)
#else // defined(DYND_ABI_COMPILING)
#define DYND_ABI_EXPORT __declspec(dllimport)
#endif // defined(DYND_ABI_COMPILING)
#elif defined(__GNUC__) // defined(_WIN32) || defined(__CYGWIN__)
// gcc and icc both still define __GNUC__,
// so this branch covers all known non-Windows compilers.
#define DYND_ABI_EXPORT __attribute__ ((visibility ("default")))
#else // defined(_WIN32) || defined(__CYGWIN__)
#error Unrecognized compiler. Unable to configure symbol visibility.
#endif // defined(_WIN32) || defined(__CYGWIN__)

#endif // !defined(DYND_ABI_VISIBILITY_H)
