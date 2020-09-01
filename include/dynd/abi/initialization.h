#if !defined(DYND_ABI_INITIALIZATION_H)
#define DYND_ABI_INITIALIZATION_H

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

#if !defined(__cplusplus)
#define dynd_default_nullptr
#elif __cplusplus >= 201103L
#define dynd_default_nullptr = nullptr
#else
#define dynd_default_nullptr = NULL
#endif

// Compiler-agnostic registration for initialization and finalization functions.
// Adapted from: https://stackoverflow.com/a/2390626.
// Initializer/finalizer sample for MSVC and GCC/Clang.
// 2010-2016 Joe Lowe. Released into the public domain.

#ifdef __cplusplus
#define DYND_INITIALIZER(f)                                  \
static void f(void);                                         \
struct f##_t_ { f##_t_(void) { f(); } }; static f##_t_ f##_; \
static void f(void)
#elif defined(_MSC_VER)
#pragma section(".CRT$XCU",read)
#define DYND_INITIALIZER2_(f,p)                          \
static void f(void);                                     \
__declspec(allocate(".CRT$XCU")) void (*f##_)(void) = f; \
__pragma(comment(linker,"/include:" p #f "_"))           \
static void f(void)
#ifdef _WIN64
#define INITIALIZER(f) INITIALIZER2_(f,"")
#else
#define INITIALIZER(f) INITIALIZER2_(f,"_")
#endif
#else
#define INITIALIZER(f) \
static void f(void) __attribute__((constructor)); \
static void f(void)
#endif

#endif // !defined(DYND_ABI_INITIALIZATION_H)
