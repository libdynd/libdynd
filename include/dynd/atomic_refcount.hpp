//
// Copyright (C) 2011-14 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

//
// Portions taken from boost (http://www.boost.org)
//
//  Copyright 2007 Peter Dimov
//
//  Distributed under the Boost Software License, Version 1.0. (See
//  http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef _DYND__ATOMIC_REFCOUNT_HPP_
#define _DYND__ATOMIC_REFCOUNT_HPP_

#include <dynd/config.hpp>

#ifdef DYND_USE_STD_ATOMIC
#include <atomic>
namespace dynd {
    typedef std::atomic<int32_t> atomic_refcount;
} // namespace dynd
#elif defined(_WIN32)

#if defined( __CLRCALL_PURE_OR_CDECL )
extern "C" long __CLRCALL_PURE_OR_CDECL _InterlockedIncrement( long volatile * );
extern "C" long __CLRCALL_PURE_OR_CDECL _InterlockedDecrement( long volatile * );
#else
extern "C" long __cdecl _InterlockedIncrement( long volatile * );
extern "C" long __cdecl _InterlockedDecrement( long volatile * );
#endif

#pragma intrinsic(_InterlockedIncrement)
#pragma intrinsic(_InterlockedDecrement)

namespace dynd {
    class atomic_refcount {
        int32_t m_refcount;

        atomic_refcount(const atomic_refcount&);
        atomic_refcount& operator=(const atomic_refcount&);
    public:
        explicit atomic_refcount(uint32_t val)
            : m_refcount(val)
        {
        }

        int32_t operator++()
        {
            return _InterlockedIncrement((long *)&m_refcount);
        }

        int32_t operator--()
        {
            return _InterlockedDecrement((long *)&m_refcount);
        }

        operator int32_t() const
        {
            return static_cast<const volatile int32_t&>(m_refcount);
        }

        bool operator!=(int32_t rhs) const {
            return static_cast<const volatile int32_t&>(m_refcount) != rhs;
        }

        bool operator<=(int32_t rhs) const {
            return static_cast<const volatile int32_t&>(m_refcount) <= rhs;
        }
    };
} // namespace dynd
#else
//  atomic_count for g++ on 486+/AMD64

namespace dynd {
    class atomic_refcount {
        int32_t m_refcount;

        atomic_refcount(const atomic_refcount&);
        atomic_refcount& operator=(const atomic_refcount&);
    public:
        explicit atomic_refcount(uint32_t val)
            : m_refcount(val)
        {
        }

        int32_t operator++()
        {
            return atomic_exchange_and_add(&m_refcount, 1) + 1;
        }

        int32_t operator--()
        {
            return atomic_exchange_and_add(&m_refcount, -1) - 1;
        }

        operator int32_t() const
        {
            return atomic_exchange_and_add((int32_t *)&m_refcount, 0);
        }
    private:

        static int atomic_exchange_and_add(int32_t * pw, int32_t dv)
        {
            // int32_t r = *pw;
            // *pw += dv;
            // return r;

            int r;

            __asm__ __volatile__
            (
                "lock\n\t"
                "xadd %1, %0":
                "+m"( *pw ), "=r"( r ): // outputs (%0, %1)
                "1"( dv ): // inputs (%2 == %1)
                "memory", "cc" // clobbers
            );

            return r;
        }
    };
} // namespace dynd
#endif

#endif // _DYND__ATOMIC_REFCOUNT_HPP_
