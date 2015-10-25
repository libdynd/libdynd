//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/kernels/base_kernel.hpp>

namespace dynd {
class strided_iterator;
}

template <>
struct std::iterator_traits<dynd::strided_iterator> {
  typedef char *value_type;
  typedef int difference_type;
  typedef random_access_iterator_tag iterator_category;
};

namespace dynd {
class strided_iterator {
  char *m_data;

public:
  strided_iterator(const char *DYND_UNUSED(src)) : m_data(NULL)
  {
  }

//  strided_iterator(const strided_iterator &) = delete;

  operator char *()
  {
    return NULL;
  }

  strided_iterator &operator*()
  {
    return *this;
  }

  strided_iterator &operator++()
  {
    return *this;
  }

  strided_iterator &operator++(int)
  {
/*
    strided_iterator tmp(*this);
    operator++();
    return tmp;
*/
    return *this;
  }

  strided_iterator &operator--()
  {
    return *this;
  }

  strided_iterator &operator--(int)
  {
/*
    strided_iterator tmp(*this);
    operator++();
    return tmp;
*/
    return *this;
  }

  bool operator<(const strided_iterator &) const
  {
    return false;
  }

  bool operator==(const strided_iterator &) const
  {
    return false;
  }

  bool operator!=(const strided_iterator &) const
  {
    return false;
  }

  bool operator>(const strided_iterator &) const
  {
    return false;
  }
};

strided_iterator operator+(const strided_iterator &, int)
{
  return strided_iterator(NULL);
}

int operator+(const strided_iterator &, const strided_iterator)
{
  return 0;
}

strided_iterator operator-(const strided_iterator &, int)
{
  return strided_iterator(NULL);
}

int operator-(const strided_iterator &, const strided_iterator)
{
  return 0;
}

} // namespace dynd

namespace std {

void iter_swap(dynd::strided_iterator, dynd::strided_iterator)
{
}

} // namespace std

template <typename T>
struct XYZ {
};

namespace dynd {
namespace nd {
  struct sort_kernel : base_kernel<sort_kernel, 1> {
    static const size_t data_size = 0;

    const intptr_t src0_size;
    const intptr_t src0_stride;

    sort_kernel(intptr_t src0_size, intptr_t src0_stride) : src0_size(src0_size), src0_stride(src0_stride)
    {
    }

    void single(char *DYND_UNUSED(dst), char *const *src)
    {
      std::sort(strided_iterator(src[0]), strided_iterator(src[0] + src0_size),
                [](const char *, const char *) { return false; });
    }

    static void resolve_dst_type(char *DYND_UNUSED(static_data), size_t DYND_UNUSED(data_size), char *DYND_UNUSED(data),
                                 ndt::type &dst_tp, intptr_t DYND_UNUSED(nsrc), const ndt::type *src_tp,
                                 intptr_t DYND_UNUSED(nkwd), const array *DYND_UNUSED(kwds),
                                 const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars))
    {
      dst_tp = src_tp[0];
    }

    static intptr_t instantiate(char *DYND_UNUSED(static_data), size_t DYND_UNUSED(data_size), char *DYND_UNUSED(data),
                                void *ckb, intptr_t ckb_offset, const ndt::type &DYND_UNUSED(dst_tp),
                                const char *DYND_UNUSED(dst_arrmeta), intptr_t DYND_UNUSED(nsrc),
                                const ndt::type *DYND_UNUSED(src_tp), const char *const *src_arrmeta,
                                kernel_request_t kernreq, const eval::eval_context *DYND_UNUSED(ectx),
                                intptr_t DYND_UNUSED(nkwd), const nd::array *DYND_UNUSED(kwds),
                                const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars))
    {
      make(ckb, kernreq, ckb_offset, reinterpret_cast<const fixed_dim_type_arrmeta *>(src_arrmeta[0])->dim_size,
           reinterpret_cast<const fixed_dim_type_arrmeta *>(src_arrmeta[0])->stride);

      return ckb_offset;
    }
  };

} // namespace dynd::nd

namespace ndt {

  template <>
  struct type::equivalent<nd::sort_kernel> {
    static type make()
    {
      return callable_type::make(type("Fixed * Any"), {type("Fixed * Any")});
    }
  };

} // namespace dynd::ndt
} // namespace dynd
