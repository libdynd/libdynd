#ifndef DYND_ABI_NOEXCEPT_H
#define DYND_ABI_NOEXCEPT_H

#if !defined(__cplusplus)
#define DYND_NOEXCEPT
#elif __cplusplus < 201103L
#define DYND_NOEXCEPT throw()
#else
#define DYND_NOEXCEPT noexcept
#endif

#endif //DYND_ABI_NOEXCEPT_H
