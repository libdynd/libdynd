#if !defined(DYND_ABI_INTEGERS_H)
#define DYND_ABI_INTEGERS_H

#if !defined(__cplusplus)
#include <stddef.h>
typedef size_t DYND_SIZE_T;
#else // !defined(__cplusplus)
#include <cstddef>
using DYND_SIZE_T = std::size_t;
#endif // !defined(__cplusplus)

#endif // !defined(DYND_ABI_INTEGERS_H)
