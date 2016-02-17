//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

////////////////////////////////////////////////////////////
// String algorithms

namespace dynd {
  namespace detail {

    template<class SelfType, class StringType>
    struct string_search_base {
      void bloom_add(uint64_t &mask, const char ch)
      {
        mask |= static_cast<uint64_t>(1) << ((ch) & (64 - 1));
      }

      uint64_t bloom(const uint64_t &mask, const char ch)
      {
        return mask & (static_cast<uint64_t>(1) << ((ch) & (64 - 1)));
      }

      /* Special case path where the needle is only a single character.
         According to effbot's numbers, for haystacks < 10 characters, a
         naive loop is fastest.  Larger than that, using POSIX `memchr` is
         faster. */
      void find_1char(const char *haystack, size_t n, char needle)
      {
        SelfType *self = (SelfType *)this;

        if (n <= 10) {
          for (size_t i = 0; i < n; ++i) {
            if (haystack[i] == needle) {
              if (self->handle_match(i)) {
                return;
              }
            }
          }
        } else {
          const char *s = haystack;
          while (s < haystack + n) {
            void *candidate = memchr((void *)s, needle, n);
            if (candidate == NULL) {
              return;
            }
            s = (const char *)candidate;
            if (self->handle_match(s - haystack)) {
              return;
            }
            s++;
          }
        }
      }

      void operator()(const StringType &haystack,
                      const StringType &needle)
      {
        /*
          This is a mostly direct copy of the algorithm by Fredrik Lundh in
          CPython, as found here:

          http://hg.python.org/cpython/file/3.5/Objects/stringlib/fastsearch.h

          and described here:

          http://effbot.org/zone/stringlib.htm

          The main differences are a result of handling UTF-8 only, and not
          three different char widths as in Python.

          There are probably further optimizations possible here, given that
          this is UTF-8. For example, we could skip over multi-byte
          sequences when a match fails, but this doesn't currently do that.
        */
        SelfType *self = (SelfType *)this;

        const char *s = haystack.begin();
        const char *p = needle.begin();
        size_t n = haystack.size();
        size_t m = needle.size();

        intptr_t w = n - m;
        if (w < 0) {
          return;
        }

        /* look for special cases */
        if (m <= 1) {
          if (m == 0) {
            return;
          }

          find_1char(haystack.begin(), n, needle.begin()[0]);
          return;
        }

        intptr_t mlast = m - 1;
        intptr_t skip = mlast - 1;
        uint64_t mask = 0;

        const char *ss = s + m - 1;
        const char *pp = p + m - 1;

        intptr_t i;
        intptr_t j;

        /* create compressed boyer-moore delta 1 table */

        /* process pattern[:-1] */
        for (i = 0; i < mlast; i++) {
          bloom_add(mask, p[i]);
          if (p[i] == p[mlast]) {
            skip = mlast - i - 1;
          }
        }

        /* process pattern[-1] outside the loop */
        bloom_add(mask, p[mlast]);

        for (i = 0; i <= w; i++) {
          /* note: using mlast in the skip path slows things down on x86 */
          if (ss[i] == pp[0]) {
            /* candidate match */
            for (j = 0; j < mlast; j++) {
              if (s[i+j] != p[j]) {
                break;
              }
            }
            if (j == mlast) {
              /* got a match! */
              if (self->handle_match(i)) {
                return;
              }
              i = i + mlast;
            }
            /* miss: check if next character is part of pattern */
            if (i < w && !bloom(mask, ss[i+1])) {
              i = i + m;
            } else {
              i = i + skip;
            }
          } else {
            /* skip: check if next character is part of pattern */
            if (i < w && !bloom(mask, ss[i+1])) {
              i = i + m;
            }
          }
        }

        return;
      }
    };


    template<class StringType>
    struct string_finder :
      public string_search_base<string_finder<StringType>, StringType>
    {
      intptr_t m_result;

      string_finder() : m_result(-1) { }

      bool handle_match(const size_t match) {
        m_result = (intptr_t)match;
        return true;
      }

      intptr_t finish() { return m_result; }
    };


    template<class StringType>
    struct string_counter :
      public string_search_base<string_counter<StringType>, StringType>
    {
      size_t m_count;

      string_counter() : m_count(0) { }

      bool handle_match(const size_t DYND_UNUSED(match)) {
        m_count++;
        return false;
      }

      intptr_t finish() { return m_count; }
    };


    template<class StringType>
    struct string_inplace_replacer :
      public string_search_base<string_inplace_replacer<StringType>, StringType>
    {
      StringType &m_dst;
      const StringType &m_new_str;

      string_inplace_replacer(StringType &dst,
                              const StringType &new_str) :
        m_dst(dst),
        m_new_str(new_str) { }

      bool handle_match(const size_t match) {
        DYND_MEMCPY(m_dst.begin() + match, m_new_str.begin(), m_new_str.size());
        return false;
      }
    };


    template<class StringType>
    struct string_copy_replacer :
      public string_search_base<string_copy_replacer<StringType>, StringType>
    {
      char *m_dst;
      const char *m_src;
      size_t m_src_size;
      size_t m_last_src_start;
      size_t m_old_str_size;
      const char *m_new_str;
      size_t m_new_str_size;

      string_copy_replacer(StringType &dst,
                           const StringType &src,
                           const StringType &old_str,
                           const StringType &new_str) :
        m_dst(dst.begin()),
        m_src(src.begin()),
        m_src_size(src.size()),
        m_last_src_start(0),
        m_old_str_size(old_str.size()),
        m_new_str(new_str.begin()),
        m_new_str_size(new_str.size())
      { }

      bool handle_match(const size_t match) {
        size_t src_chunk_size = match - m_last_src_start;

        DYND_MEMCPY(m_dst, m_src + m_last_src_start, src_chunk_size);
        m_dst += src_chunk_size;
        DYND_MEMCPY(m_dst, m_new_str, m_new_str_size);
        m_dst += m_new_str_size;
        m_last_src_start = match + m_old_str_size;

        return false;
      }

      void finish() {
        DYND_MEMCPY(m_dst, m_src + m_last_src_start, m_src_size - m_last_src_start);
      }
    };

    template<class StringType>
    struct string_splitter :
      public string_search_base<string_splitter<StringType>, StringType>
    {
      StringType *m_dst;
      const char *m_src;
      size_t m_src_size;
      size_t m_n;
      size_t m_i;
      size_t m_last_src_start;
      size_t m_split_size;

      string_splitter(StringType *dst, const StringType &src, const StringType& split) :
        m_dst(dst),
        m_src(src.begin()),
        m_src_size(src.size()),
        m_i(0),
        m_last_src_start(0),
        m_split_size(split.size())
      {

      }

      bool handle_match(const size_t match) {
        size_t new_size = match - m_last_src_start;

        new (&m_dst[m_i]) StringType(m_src + m_last_src_start, new_size);
        m_last_src_start += new_size + m_split_size;
        m_i++;

        return false;
      }

      void finish() {
        size_t new_size = m_src_size - m_last_src_start;

        new (&m_dst[m_i]) StringType(m_src + m_last_src_start, new_size);
      }
    };

  } // namespace detail
} // namespace nd
