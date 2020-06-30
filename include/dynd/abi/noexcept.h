#if !defined(DYND_ABI_NOEXCEPT_H)
#define DYND_ABI_NOEXCEPT_H

#if !defined(__cplusplus)
#define DYND_NOEXCEPT
#elif __cplusplus < 201103L // !defined(__cplusplus)
#define DYND_NOEXCEPT throw()
#else // !defined(__cplusplus)
#define DYND_NOEXCEPT noexcept
#endif

#endif // !defined(DYND_ABI_NOEXCEPT_H)
