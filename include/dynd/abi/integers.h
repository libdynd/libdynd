#if !defined(DYND_ABI_INTEGERS_H)
#define DYND_ABI_INTEGERS_H

#if !defined(__cplusplus)
#include <stddef.h>
typedef size_t dynd_size_t;
#else // !defined(__cplusplus)
#include <cstddef>
using dynd_size_t = std::size_t;
#endif // !defined(__cplusplus)

#endif // !defined(DYND_ABI_INTEGERS_H)
