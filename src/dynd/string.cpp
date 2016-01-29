//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/string.hpp>
#include <dynd/kernels/string_concat_kernel.hpp>
#include <dynd/kernels/string_count_kernel.hpp>
#include <dynd/kernels/string_find_kernel.hpp>
#include <dynd/kernels/string_replace_kernel.hpp>
#include <dynd/func/elwise.hpp>

using namespace std;
using namespace dynd;


////////////////////////////////////////////////////////////
// String algorithms


void dynd::string_concat(size_t nop, dynd::string *d, const dynd::string *const *s)
{
  // Get the size of the concatenated string
  size_t size = 0;
  for (size_t i = 0; i != nop; ++i) {
    size += (s[i]->end() - s[i]->begin());
  }

  // Allocate the output
  d->resize(size);
  // Copy the string data
  char *dst = d->begin();
  for (size_t i = 0; i != nop; ++i) {
    size_t op_size = (s[i]->end() - s[i]->begin());
    DYND_MEMCPY(dst, s[i]->begin(), op_size);
    dst += op_size;
  }
}


template<class SelfType>
struct string_search_base {
  inline void bloom_add(uint64_t &mask, const char ch)
  {
    mask |= 1UL << ((ch) & (64 - 1));
  }

  inline uint64_t bloom(const uint64_t &mask, const char ch)
  {
    return mask & (1UL << ((ch) & (64 - 1)));
  }

  /* Special case path where the needle is only a single character.
     According to effbot's numbers, for haystacks < 10 characters, a
     naive loop is fastest.  Larger than that, using POSIX `memchr` is
     faster. */
  inline void find_1char(const char *haystack, size_t n, char needle)
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

  void operator()(const dynd::string *const haystack,
                  const dynd::string *const needle)
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

    const char *s = haystack->begin();
    const char *p = needle->begin();
    size_t n = haystack->size();
    size_t m = needle->size();

    size_t w = n - m;
    if (w < 0) {
      return;
    }

    /* look for special cases */
    if (m <= 1) {
      if (m == 0) {
        return;
      }

      find_1char(haystack->begin(), n, needle->begin()[0]);
      return;
    }

    size_t mlast = m - 1;
    size_t skip = mlast - 1;
    uint64_t mask = 0;

    const char *ss = s + m - 1;
    const char *pp = p + m - 1;

    size_t i;
    size_t j;

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
        if (!bloom(mask, ss[i+1])) {
          i = i + m;
        } else {
          i = i + skip;
        }
      } else {
        /* skip: check if next character is part of pattern */
        if (!bloom(mask, ss[i+1])) {
          i = i + m;
        }
      }
    }

    return;
  }
};


struct string_finder : public string_search_base<string_finder>
{
  intptr_t m_result;

  string_finder() : m_result(-1) { }

  bool handle_match(const size_t match) {
    m_result = (intptr_t)match;
    return true;
  }
};


intptr_t dynd::string_find(const dynd::string *const haystack,
                           const dynd::string *const needle) {
  string_finder f;

  f(haystack, needle);

  return f.m_result;
}


struct string_counter : public string_search_base<string_counter>
{
  size_t m_count;

  string_counter() : m_count(0) { }

  bool handle_match(const size_t DYND_UNUSED(match)) {
    m_count++;
    return false;
  }
};


intptr_t dynd::string_count(const dynd::string *const haystack,
                            const dynd::string *const needle) {
  string_counter f;

  f(haystack, needle);

  return f.m_count;
}


struct string_inplace_replacer : public string_search_base<string_inplace_replacer>
{
  dynd::string *m_dst;
  const dynd::string *m_new_str;

  string_inplace_replacer(dynd::string *const dst,
                          const dynd::string *const new_str) :
    m_dst(dst),
    m_new_str(new_str) { }

  bool handle_match(const size_t match) {
    memcpy(m_dst->begin() + match, m_new_str->begin(), m_new_str->size());
    return false;
  }
};


struct string_copy_replacer : public string_search_base<string_copy_replacer>
{
  char *m_dst;
  const char *m_src;
  size_t m_src_size;
  size_t m_last_src_start;
  size_t m_old_str_size;
  const char *m_new_str;
  size_t m_new_str_size;

  string_copy_replacer(dynd::string *const dst,
                       const dynd::string *const src,
                       const dynd::string *const old_str,
                       const dynd::string *const new_str) :
    m_dst(dst->begin()),
    m_src(src->begin()),
    m_src_size(src->size()),
    m_last_src_start(0),
    m_old_str_size(old_str->size()),
    m_new_str(new_str->begin()),
    m_new_str_size(new_str->size())
  { }

  bool handle_match(const size_t match) {
    size_t src_chunk_size = match - m_last_src_start;

    memcpy(m_dst, m_src + m_last_src_start, src_chunk_size);
    m_dst += src_chunk_size;
    memcpy(m_dst, m_new_str, m_new_str_size);
    m_dst += m_new_str_size;
    m_last_src_start = match + m_old_str_size;

    return false;
  }

  void handle_end() {
    memcpy(m_dst, m_src + m_last_src_start, m_src_size - m_last_src_start);
  }
};


void dynd::string_replace(dynd::string *const dst,
                          const dynd::string *const src,
                          const dynd::string *const old_str,
                          const dynd::string *const new_str) {

  if (old_str->size() == 0) {
    /* Just copy -- there's nothing to replace */
    *dst = *src;
  } else if (old_str->size() == new_str->size()) {
    /* Special case when old_str and new_str are same length,
       we copy src to dst and the replace in-place. */
    *dst = *src;

    if (old_str->size() == 1) {
      /* Special case when old_str and new_str are both 1 character */
      char old_chr = old_str->begin()[0];
      char new_chr = new_str->begin()[0];
      for (auto p = dst->begin(); p != dst->end(); ++p) {
        if (*p == old_chr) {
          *p = new_chr;
        }
      }
    } else {
      string_inplace_replacer replacer(dst, new_str);
      replacer(src, old_str);
    }
  } else {
    /* Most general case, where old_str and new_str are different
       lengths.  Count matches to determine resulting string length,
       then interleave to make the result. */
    intptr_t count = string_count(src, old_str);
    size_t delta =
      ((intptr_t)new_str->size() - (intptr_t)old_str->size()) * count;

    dst->resize((intptr_t)src->size() + delta);

    string_copy_replacer replacer(dst, src, old_str, new_str);
    replacer(src, old_str);
    replacer.handle_end();
  }
}


////////////////////////////////////////////////////////////
// String kernels


DYND_API nd::callable nd::string_concatenation::make()
{
  return nd::functional::elwise(nd::callable::make<nd::string_concatenation_kernel>());
}

DYND_API struct nd::string_concatenation nd::string_concatenation;

DYND_API nd::callable nd::string_count::make()
{
  return nd::functional::elwise(nd::callable::make<nd::string_count_kernel>());
}

DYND_API struct nd::string_count nd::string_count;

DYND_API nd::callable nd::string_find::make()
{
  return nd::functional::elwise(nd::callable::make<nd::string_find_kernel>());
}

DYND_API struct nd::string_find nd::string_find;

DYND_API nd::callable nd::string_replace::make()
{
  return nd::functional::elwise(nd::callable::make<nd::string_replace_kernel>());
}

DYND_API struct nd::string_replace nd::string_replace;
