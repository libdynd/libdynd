//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/kernels/base_kernel.hpp>

namespace dynd {
namespace nd {

  struct sort_kernel : base_kernel<sort_kernel, 1> {
    class bytes;
    class iterator;

    static const size_t data_size = 0;

    const intptr_t src0_size;
    const intptr_t src0_stride;

    sort_kernel(intptr_t src0_size, intptr_t src0_stride) : src0_size(src0_size), src0_stride(src0_stride)
    {
    }

    void single(char *DYND_UNUSED(dst), char *const *src);

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
} // namespace DyND

template <>
struct std::iterator_traits<dynd::nd::sort_kernel::iterator> {
  typedef dynd::nd::sort_kernel::bytes value_type;
  typedef int difference_type;
  typedef random_access_iterator_tag iterator_category;
};

namespace dynd {
namespace nd {

  class sort_kernel::bytes {
  protected:
    char *m_data;
    size_t m_size;
    bool dealloc;

  public:
    bytes(char *data, size_t size) : m_data(data), m_size(size), dealloc(false)
    {
    }

    bytes(const bytes &other) : dealloc(true)
    {
      m_data = reinterpret_cast<char *>(malloc(other.m_size));
      m_size = other.m_size;
      std::memcpy(m_data, other.m_data, other.m_size);
    }

    ~bytes()
    {
      if (dealloc) {
        free(m_data);
      }
    }

    char *data()
    {
      return m_data;
    }

    const char *data() const
    {
      return m_data;
    }

    size_t size() const
    {
      return m_size;
    }

    bytes &operator=(const bytes &rhs)
    {
      std::memcpy(m_data, rhs.m_data, rhs.m_size);
      m_size = rhs.m_size;

      return *this;
    }
  };

  class sort_kernel::iterator : public bytes {
    intptr_t m_stride;

  public:
    iterator(char *data, size_t size) : bytes(data, size), m_stride(sizeof(int))
    {
    }

    iterator(const iterator &other) : bytes(other.m_data, other.m_size), m_stride(other.m_stride)
    {
    }

    intptr_t stride() const
    {
      return m_stride;
    }

    bytes &operator*()
    {
      return *this;
    }

    iterator &operator++()
    {
      m_data += m_stride;
      return *this;
    }

    iterator operator++(int)
    {
      iterator tmp(*this);
      operator++();
      return tmp;
    }

    iterator &operator--()
    {
      m_data -= m_stride;
      return *this;
    }

    iterator operator--(int)
    {
      iterator tmp(*this);
      operator--();
      return tmp;
    }

    bool operator<(const iterator &rhs) const
    {
      return m_data < rhs.m_data;
    }

    bool operator==(const iterator &rhs) const
    {
      return m_data == rhs.m_data;
    }

    bool operator!=(const iterator &rhs) const
    {
      return m_data != rhs.m_data;
    }

    bool operator>(const iterator &rhs) const
    {
      return m_data > rhs.m_data;
    }

    int operator-(iterator rhs)
    {
      return (m_data - rhs.m_data) / m_stride;
    }
  };

  sort_kernel::iterator operator+(sort_kernel::iterator lhs, int rhs)
  {
    return sort_kernel::iterator(lhs.data() + rhs * lhs.stride(), lhs.size());
  }

  sort_kernel::iterator operator-(sort_kernel::iterator lhs, int rhs)
  {
    return sort_kernel::iterator(lhs.data() - rhs * lhs.stride(), lhs.size());
  }

  void sort_kernel::single(char *DYND_UNUSED(dst), char *const *src)
  {
    std::cout << "begin = " << reinterpret_cast<intptr_t>(src[0]) << std::endl;
    std::cout << "end = " << reinterpret_cast<intptr_t>(src[0] + src0_size * sizeof(int)) << std::endl;

    std::sort(iterator(src[0], sizeof(int)), iterator(src[0] + src0_size * sizeof(int), sizeof(int)),
              [](const bytes &lhs, const bytes &rhs) {
      std::cout << "comparing" << std::endl;
      std::cout << "lhs = " << *reinterpret_cast<const int *>(lhs.data()) << std::endl;
      std::cout << "rhs = " << *reinterpret_cast<const int *>(rhs.data()) << std::endl;
      return *reinterpret_cast<const int *>(lhs.data()) < *reinterpret_cast<const int *>(rhs.data());
    });
  }

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
