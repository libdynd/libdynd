#if !defined(DYND_ABI_NOEXCEPT_H)
#define DYND_ABI_NOEXCEPT_H

#if !defined(__cplusplus)
#define dynd_noexcept
#elif __cplusplus < 201103L // !defined(__cplusplus)
#define dynd_noexcept throw()
#else // !defined(__cplusplus)
#define dynd_noexcept noexcept
#endif

#endif // !defined(DYND_ABI_NOEXCEPT_H)
