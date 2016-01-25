//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <new>
#include <algorithm>
#include <map>

#include <dynd/kernels/ckernel_prefix.hpp>

namespace dynd {
namespace nd {

  /**
   * Increments a ``ckb_offset`` variable (offset into a ckernel_builder)
   * by the provided increment. The increment needs to be aligned to 8 bytes,
   * so padding may be added.
   */
  inline void inc_ckb_offset(intptr_t &inout_ckb_offset, size_t inc)
  {
    inout_ckb_offset += static_cast<intptr_t>(ckernel_prefix::align_offset(inc));
  }

  template <class T = ckernel_prefix>
  void inc_ckb_offset(intptr_t &inout_ckb_offset)
  {
    inc_ckb_offset(inout_ckb_offset, sizeof(T));
  }

  /**
   * Function pointers + data for a hierarchical
   * kernel which operates on type/arrmeta in
   * some configuration.
   *
   * The data placed in the kernel's data must
   * be relocatable with a memcpy, it must not rely on its
   * own address.
   */
  class kernel_builder {
  public:
    intptr_t m_size;

  protected:
    // Pointer to the kernel function pointers + data
    char *m_data;
    intptr_t m_capacity;

    // When the amount of data is small, this static data is used,
    // otherwise dynamic memory is allocated when it gets too big
    char m_static_data[16 * 8];

    bool using_static_data() const { return m_data == &m_static_data[0]; }

    void init()
    {
      m_data = &m_static_data[0];
      m_size = 0;
      m_capacity = sizeof(m_static_data);
      set(m_static_data, 0, sizeof(m_static_data));
    }

    void destroy()
    {
      if (m_data != NULL) {
        // Destroy whatever was created
        destroy(reinterpret_cast<ckernel_prefix *>(m_data));
        // Free the memory
        free(m_data);
      }
    }

  public:
    kernel_builder() { init(); }

    ~kernel_builder() { destroy(); }

    template <typename SelfType, typename PrefixType, typename... A>
    SelfType *init(PrefixType *rawself, kernel_request_t kernreq, A &&... args)
    {
      /* Alignment requirement of the type. */
      static_assert(alignof(SelfType) <= alignof(uint64_t), "ckernel types require alignment <= 64 bits");

      return SelfType::init(rawself, kernreq, std::forward<A>(args)...);
    }

    void destroy(ckernel_prefix *self) { self->destroy(); }

    void reset()
    {
      destroy();
      init();
    }

    /**
     * This function ensures that the ckernel's data
     * is at least the required number of bytes. It
     * should only be called during the construction phase
     * of the kernel when constructing a leaf kernel.
     */
    void reserve(intptr_t requested_capacity)
    {
      if (m_capacity < requested_capacity) {
        // Grow by a factor of 1.5
        // https://github.com/facebook/folly/blob/master/folly/docs/FBVector.md
        intptr_t grown_capacity = m_capacity * 3 / 2;
        if (requested_capacity < grown_capacity) {
          requested_capacity = grown_capacity;
        }
        // Do a realloc
        char *new_data = reinterpret_cast<char *>(realloc(m_data, m_capacity, requested_capacity));
        if (new_data == NULL) {
          destroy();
          m_data = NULL;
          throw std::bad_alloc();
        }
        // Zero out the newly allocated capacity
        set(reinterpret_cast<char *>(new_data) + m_capacity, 0, requested_capacity - m_capacity);
        m_data = new_data;
        m_capacity = requested_capacity;
      }
    }

    /**
     * For use during construction. This function ensures that the
     * ckernel_builder has enough capacity (including a child), increments the
     * provided offset appropriately based on the size of T, and returns a pointer
     * to the allocated ckernel.
     */
    template <class T>
    T *alloc_ck()
    {
      intptr_t ckb_offset = m_size;
      inc_ckb_offset<T>(m_size);
      reserve(m_size);
      return reinterpret_cast<T *>(m_data + ckb_offset);
    }

    ckernel_prefix *get() const { return reinterpret_cast<ckernel_prefix *>(m_data); }

    /**
     * For use during construction, gets the ckernel component
     * at the requested offset.
     */
    template <typename KernelType>
    KernelType *get_at(size_t offset)
    {
      return reinterpret_cast<KernelType *>(m_data + offset);
    }

    void *alloc(size_t size) { return std::malloc(size); }

    void *realloc(void *ptr, size_t old_size, size_t new_size)
    {
      if (using_static_data()) {
        // If we were previously using the static data, do a malloc
        void *new_data = alloc(new_size);
        // If the allocation succeeded, copy the old data as the realloc would
        if (new_data != NULL) {
          copy(new_data, ptr, old_size);
        }
        return new_data;
      }
      else {
        return std::realloc(ptr, new_size);
      }
    }

    void free(void *ptr)
    {
      if (!using_static_data()) {
        std::free(ptr);
      }
    }

    void *copy(void *dst, const void *src, size_t size) { return std::memcpy(dst, src, size); }

    void *set(void *dst, int value, size_t size) { return std::memset(dst, value, size); }

    /** For debugging/informational purposes */
    intptr_t get_capacity() const { return m_capacity; }

    /**
     * Creates the kernel, and increments ``m_size`` to the position after it.
     */
    template <typename KernelType, typename... ArgTypes>
    void emplace_back(kernel_request_t kernreq, ArgTypes &&... args)
    {
      intptr_t ckb_offset = m_size;
      inc_ckb_offset<KernelType>(m_size);
      reserve(m_size);
      KernelType *rawself = this->get_at<KernelType>(ckb_offset);
      this->init<KernelType>(rawself, kernreq, std::forward<ArgTypes>(args)...);
    }
  };

} // namespace dynd::nd

inline void ckernel_prefix::instantiate(char *static_data, char *DYND_UNUSED(data), nd::kernel_builder *ckb,
                                        intptr_t DYND_UNUSED(ckb_offset), const ndt::type &DYND_UNUSED(dst_tp),
                                        const char *DYND_UNUSED(dst_arrmeta), intptr_t DYND_UNUSED(nsrc),
                                        const ndt::type *DYND_UNUSED(src_tp),
                                        const char *const *DYND_UNUSED(src_arrmeta), kernel_request_t kernreq,
                                        intptr_t DYND_UNUSED(nkwd), const nd::array *DYND_UNUSED(kwds),
                                        const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars))
{
  void *func;
  switch (kernreq) {
  case kernel_request_single:
    func = reinterpret_cast<kernel_targets_t *>(static_data)->single;
    break;
  case kernel_request_strided:
    func = reinterpret_cast<kernel_targets_t *>(static_data)->strided;
    break;
  default:
    throw std::invalid_argument("unrecognized kernel request");
    break;
  }

  if (func == NULL) {
    throw std::invalid_argument("no kernel request");
  }

  ckb->emplace_back<ckernel_prefix>(kernreq, func);
}

} // namespace dynd
