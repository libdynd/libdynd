//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/bytes.hpp>
#include <dynd/func/comparison.hpp>
#include <dynd/kernels/base_kernel.hpp>

namespace dynd {

class bytes_iterator : public bytes {
  intptr_t m_stride;

public:
  bytes_iterator(char *data, size_t size, intptr_t stride) : m_stride(stride)
  {
    m_data = data;
    m_size = size;
  }

  bytes_iterator(const bytes_iterator &other) : m_stride(other.m_stride)
  {
    m_data = other.m_data;
    m_size = other.m_size;
  }

  ~bytes_iterator()
  {
    m_data = NULL;
    m_size = 0;
  }

  intptr_t stride() const
  {
    return m_stride;
  }

  bytes &operator*()
  {
    return *this;
  }

  bytes_iterator &operator++()
  {
    m_data += m_stride;
    return *this;
  }

  bytes_iterator operator++(int)
  {
    bytes_iterator tmp(*this);
    operator++();
    return tmp;
  }

  bytes_iterator &operator+=(int i)
  {
    m_data += i * m_stride;
    return *this;
  }

  bytes_iterator &operator--()
  {
    m_data -= m_stride;
    return *this;
  }

  bytes_iterator operator--(int)
  {
    bytes_iterator tmp(*this);
    operator--();
    return tmp;
  }

  bytes_iterator &operator-=(int i)
  {
    m_data -= i * m_stride;
    return *this;
  }

  bool operator<(const bytes_iterator &rhs) const
  {
    return m_data < rhs.m_data;
  }

  bool operator<=(const bytes_iterator &rhs) const
  {
    return m_data <= rhs.m_data;
  }

  bool operator==(const bytes_iterator &rhs) const
  {
    return m_data == rhs.m_data;
  }

  bool operator!=(const bytes_iterator &rhs) const
  {
    return m_data != rhs.m_data;
  }

  bool operator>=(const bytes_iterator &rhs) const
  {
    return m_data >= rhs.m_data;
  }

  bool operator>(const bytes_iterator &rhs) const
  {
    return m_data > rhs.m_data;
  }

  intptr_t operator-(bytes_iterator rhs)
  {
    return (m_data - rhs.m_data) / m_stride;
  }

  bytes_iterator &operator=(const bytes_iterator &other)
  {
    m_data = other.m_data;
    m_size = other.m_size;

    return *this;
  }
};

inline bytes_iterator operator+(bytes_iterator lhs, intptr_t rhs)
{
  return bytes_iterator(lhs.data() + rhs * lhs.stride(), lhs.size(), lhs.stride());
}

inline bytes_iterator operator-(bytes_iterator lhs, intptr_t rhs)
{
  return bytes_iterator(lhs.data() - rhs * lhs.stride(), lhs.size(), lhs.stride());
}

} // namespace dynd

namespace std {

template <>
struct iterator_traits<dynd::bytes_iterator> {
  typedef dynd::bytes value_type;
  typedef intptr_t difference_type;
  typedef random_access_iterator_tag iterator_category;
};

} // namespace std

namespace dynd {
namespace nd {

  struct sort_kernel : base_kernel<sort_kernel, 1> {
    class iterator;

    static const size_t data_size = 0;

    const intptr_t src0_size;
    const intptr_t src0_stride;
    const intptr_t src0_element_data_size;

    sort_kernel(intptr_t src0_size, intptr_t src0_stride, size_t src0_element_data_size)
        : src0_size(src0_size), src0_stride(src0_stride), src0_element_data_size(src0_element_data_size)
    {
    }

    ~sort_kernel()
    {
      get_child()->destroy();
    }

    void single(char *DYND_UNUSED(dst), char *const *src)
    {
      ckernel_prefix *child = get_child();
      std::sort(bytes_iterator(src[0], src0_element_data_size, src0_stride),
                bytes_iterator(src[0] + src0_size * src0_stride, src0_element_data_size, src0_stride),
                [child](const bytes &lhs, const bytes &rhs) {
        bool1 dst;
        char *src[2] = {const_cast<char *>(lhs.data()), const_cast<char *>(rhs.data())};
        child->single(reinterpret_cast<char *>(&dst), src);
        return dst;
      });
    }

    static intptr_t instantiate(char *DYND_UNUSED(static_data), size_t DYND_UNUSED(data_size), char *data, void *ckb,
                                intptr_t ckb_offset, const ndt::type &DYND_UNUSED(dst_tp),
                                const char *DYND_UNUSED(dst_arrmeta), intptr_t DYND_UNUSED(nsrc),
                                const ndt::type *src_tp, const char *const *src_arrmeta, kernel_request_t kernreq,
                                const eval::eval_context *ectx, intptr_t nkwd, const nd::array *kwds,
                                const std::map<std::string, ndt::type> &tp_vars)
    {
      const ndt::type &src0_element_tp = src_tp[0].template extended<ndt::fixed_dim_type>()->get_element_type();

      make(ckb, kernreq, ckb_offset, reinterpret_cast<const fixed_dim_type_arrmeta *>(src_arrmeta[0])->dim_size,
           reinterpret_cast<const fixed_dim_type_arrmeta *>(src_arrmeta[0])->stride, src0_element_tp.get_data_size());

      const ndt::type child_src_tp[2] = {src0_element_tp, src0_element_tp};
      return nd::less::get().get()->instantiate(nd::less::get().get()->static_data, nd::less::get().get()->data_size,
                                                data, ckb, ckb_offset, ndt::type::make<bool1>(), NULL, 2, child_src_tp,
                                                NULL, kernel_request_single, ectx, nkwd, kwds, tp_vars);
    }
  };

} // namespace dynd::nd

namespace ndt {

  template <>
  struct type::equivalent<nd::sort_kernel> {
    static type make()
    {
      return callable_type::make(type::make<void>(), {type("Fixed * Scalar")});
    }
  };

} // namespace dynd::ndt
} // namespace dynd
