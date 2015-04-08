//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <new>
#include <algorithm>
#include <map>

#include <dynd/config.hpp>
#include <dynd/kernels/ckernel_prefix.hpp>
#include <dynd/types/type_id.hpp>

namespace dynd {

/**
 * Increments a ``ckb_offset`` variable (offset into a ckernel_builder)
 * by the provided increment. The increment needs to be aligned to 8 bytes,
 * so padding may be added.
 */
inline void inc_ckb_offset(intptr_t &inout_ckb_offset, size_t inc)
{
  inout_ckb_offset += static_cast<intptr_t>(ckernel_prefix::align_offset(inc));
}

template <class T>
void inc_ckb_offset(intptr_t &inout_ckb_offset)
{
  inc_ckb_offset(inout_ckb_offset, sizeof(T));
}

/**
 * Function pointers + data for a hierarchical
 * kernel which operates on type/arrmeta in
 * some configuration. Individual kernel types
 * are handled by the classes unary_ckernel_builder, etc.
 *
 * The data placed in the kernel's data must
 * be relocatable with a memcpy, it must not rely on its
 * own address.
 */
template <typename CKBT>
class base_ckernel_builder {
protected:
  // Pointer to the kernel function pointers + data
  char *m_data;
  intptr_t m_capacity;

  void destroy()
  {
    if (m_data != NULL) {
      // Destroy whatever was created
      reinterpret_cast<CKBT *>(this)
          ->destroy(reinterpret_cast<ckernel_prefix *>(m_data));
      // Free the memory
      reinterpret_cast<CKBT *>(this)->free(m_data);
    }
  }

public:
  base_ckernel_builder() { reinterpret_cast<CKBT *>(this)->init(); }

  ~base_ckernel_builder() { reinterpret_cast<CKBT *>(this)->destroy(); }

  template <typename self_type, typename... A>
  self_type *init(ckernel_prefix *rawself, kernel_request_t kernreq,
                  A &&... args)
  {
    return reinterpret_cast<CKBT *>(this)
        ->template init<self_type>(rawself, kernreq, std::forward<A>(args)...);
  }

  void reset()
  {
    reinterpret_cast<CKBT *>(this)->destroy();
    reinterpret_cast<CKBT *>(this)->init();
  }

  /**
   * This function ensures that the ckernel's data
   * is at least the required number of bytes. It
   * should only be called during the construction phase
   * of the kernel.
   *
   * NOTE: This function ensures that there is room for
   *       another base at the end, so if you are sure
   *       that you're a leaf kernel, use reserve_leaf
   *       instead.
   */
  void reserve(intptr_t requested_capacity)
  {
    reserve_leaf(requested_capacity + sizeof(ckernel_prefix));
  }

  /**
   * This function ensures that the ckernel's data
   * is at least the required number of bytes. It
   * should only be called during the construction phase
   * of the kernel when constructing a leaf kernel.
   */
  void reserve_leaf(intptr_t requested_capacity)
  {
    if (m_capacity < requested_capacity) {
      // Grow by a factor of 1.5
      // https://github.com/facebook/folly/blob/master/folly/docs/FBVector.md
      intptr_t grown_capacity = m_capacity * 3 / 2;
      if (requested_capacity < grown_capacity) {
        requested_capacity = grown_capacity;
      }
      // Do a realloc
      char *new_data =
          reinterpret_cast<char *>(reinterpret_cast<CKBT *>(this)->realloc(
              m_data, m_capacity, requested_capacity));
      if (new_data == NULL) {
        reinterpret_cast<CKBT *>(this)->destroy();
        m_data = NULL;
        throw std::bad_alloc();
      }
      // Zero out the newly allocated capacity
      reinterpret_cast<CKBT *>(this)
          ->set(reinterpret_cast<char *>(new_data) + m_capacity, 0,
                requested_capacity - m_capacity);
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
  T *alloc_ck(intptr_t &inout_ckb_offset)
  {
    intptr_t ckb_offset = inout_ckb_offset;
    inc_ckb_offset<T>(inout_ckb_offset);
    reserve(inout_ckb_offset);
    return reinterpret_cast<T *>(m_data + ckb_offset);
  }

  /**
   * For use during construction. This function ensures that the
   * ckernel_builder has enough capacity, increments the provided
   * offset appropriately based on the size of T, and returns a pointer
   * to the allocated ckernel.
   */
  template <class T>
  T *alloc_ck_leaf(intptr_t &inout_ckb_offset)
  {
    intptr_t ckb_offset = inout_ckb_offset;
    inc_ckb_offset<T>(inout_ckb_offset);
    reserve(inout_ckb_offset);
    return reinterpret_cast<T *>(m_data + ckb_offset);
  }

  /**
   * For use during construction, gets the ckernel component
   * at the requested offset.
   */
  template <class T>
  T *get_at(size_t offset)
  {
    return reinterpret_cast<T *>(m_data + offset);
  }

  ckernel_prefix *get() const
  {
    return reinterpret_cast<ckernel_prefix *>(m_data);
  }

  void swap(base_ckernel_builder &rhs)
  {
    (std::swap)(m_data, rhs.m_data);
    (std::swap)(m_capacity, rhs.m_capacity);
  }

  /** For debugging/informational purposes */
  intptr_t get_capacity() const { return m_capacity; }
};

template <kernel_request_t kernreq>
class ckernel_builder;

template <>
class ckernel_builder<kernel_request_host>
    : public base_ckernel_builder<ckernel_builder<kernel_request_host>> {
  // When the amount of data is small, this static data is used,
  // otherwise dynamic memory is allocated when it gets too big
  char m_static_data[16 * 8];

  bool using_static_data() const { return m_data == &m_static_data[0]; }

public:
  void init()
  {
    m_data = &m_static_data[0];
    m_capacity = sizeof(m_static_data);
    set(m_static_data, 0, sizeof(m_static_data));
  }

  template <typename self_type, typename... A>
  self_type *init(ckernel_prefix *rawself, kernel_request_t kernreq,
                  A &&... args)
  {
    return self_type::init(rawself, kernreq, std::forward<A>(args)...);
  }

  void destroy()
  {
    base_ckernel_builder<ckernel_builder<kernel_request_host>>::destroy();
  }

  void destroy(ckernel_prefix *self) { self->destroy(); }

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
    } else {
      return std::realloc(ptr, new_size);
    }
  }

  void free(void *ptr)
  {
    if (!using_static_data()) {
      std::free(ptr);
    }
  }

  void *copy(void *dst, const void *src, size_t size)
  {
    return std::memcpy(dst, src, size);
  }

  void *set(void *dst, int value, size_t size)
  {
    return std::memset(dst, value, size);
  }

  void swap(ckernel_builder<kernel_request_host> &rhs)
  {
    if (using_static_data()) {
      if (rhs.using_static_data()) {
        char tmp_static_data[sizeof(m_static_data)];
        copy(tmp_static_data, m_static_data, sizeof(m_static_data));
        copy(m_static_data, rhs.m_static_data, sizeof(m_static_data));
        copy(rhs.m_static_data, tmp_static_data, sizeof(m_static_data));
      } else {
        copy(rhs.m_static_data, m_static_data, sizeof(m_static_data));
        m_data = rhs.m_data;
        m_capacity = rhs.m_capacity;
        rhs.m_data = &rhs.m_static_data[0];
        rhs.m_capacity = 16 * sizeof(intptr_t);
      }
    } else {
      if (rhs.using_static_data()) {
        copy(m_static_data, rhs.m_static_data, sizeof(m_static_data));
        rhs.m_data = m_data;
        rhs.m_capacity = m_capacity;
        m_data = &m_static_data[0];
        m_capacity = sizeof(m_static_data);
      } else {
        (std::swap)(m_data, rhs.m_data);
        (std::swap)(m_capacity, rhs.m_capacity);
      }
    }
  }
};

#ifdef __CUDACC__

void cuda_throw_if_not_success(cudaError_t);

template <typename self_type, typename... A>
__global__ void cuda_device_init(ckernel_prefix *rawself,
                                 kernel_request_t kernreq, A... args)
{
  self_type::init(rawself, kernreq, args...);
}

__global__ void cuda_device_destroy(ckernel_prefix *self);

template <>
class ckernel_builder<kernel_request_cuda_device>
    : public base_ckernel_builder<ckernel_builder<kernel_request_cuda_device>> {
  static class pooled_allocator {
    std::multimap<std::size_t, void *> available_blocks;
    std::map<void *, std::size_t> used_blocks;

  public:
    ~pooled_allocator()
    {
      /*
            Todo: This needs to deallocate all existing allocations. It
         currently throws
                  an exception.

            for (std::multimap<std::size_t, void *>::iterator i =
                     available_blocks.begin();
                 i != available_blocks.end(); ++i) {
              cuda_throw_if_not_success(cudaFree(i->second));
            }
            available_blocks.clear();
            for (std::map<void *, std::size_t>::iterator i =
         used_blocks.begin();
                 i != used_blocks.end(); ++i) {
              cuda_throw_if_not_success(cudaFree(i->first));
            }
            used_blocks.clear();
      */
    }

    void *allocate(size_t n)
    {
      void *res;
      std::multimap<std::size_t, void *>::iterator available_block =
          available_blocks.find(n);

      if (available_block != available_blocks.end()) {
        res = available_block->second;
        available_blocks.erase(available_block);
      } else {
        cuda_throw_if_not_success(cudaMalloc(&res, n));
      }

      used_blocks.insert(std::make_pair(res, n));

      return res;
    }

    void deallocate(void *ptr)
    {
      std::map<void *, std::size_t>::iterator iter = used_blocks.find(ptr);
      std::size_t num_bytes = iter->second;
      used_blocks.erase(iter);
      available_blocks.insert(std::make_pair(num_bytes, ptr));
    }
  } allocator;

public:
  void init()
  {
    m_data = reinterpret_cast<char *>(alloc(16 * 8));
    m_capacity = 16 * 8;
  }

  void *alloc(size_t size) { return allocator.allocate(size); }

  void *realloc(void *old_ptr, size_t old_size, size_t new_size)
  {
    void *new_ptr = alloc(new_size);
    copy(new_ptr, old_ptr, old_size);
    free(old_ptr);
    return new_ptr;
  }

  void free(void *ptr) { allocator.deallocate(ptr); }

  void *copy(void *dst, const void *src, size_t size)
  {
    cuda_throw_if_not_success(
        cudaMemcpy(dst, src, size, cudaMemcpyDeviceToDevice));
    return dst;
  }

  void *set(void *dst, int DYND_UNUSED(value), size_t DYND_UNUSED(size))
  {
    // Todo: Delete this member function.
    return dst;
  }

  template <typename self_type, typename... A>
  self_type *init(ckernel_prefix *rawself, kernel_request_t kernreq,
                  A &&... args)
  {
    cuda_device_init<self_type> << <1, 1>>>
        (rawself, kernreq, std::forward<A>(args)...);
    // check for CUDA errors here

    return self_type::get_self(rawself);
  }

  void destroy()
  {
    base_ckernel_builder<
        ckernel_builder<kernel_request_cuda_device>>::destroy();
  }

  void destroy(ckernel_prefix *self)
  {
    cuda_device_destroy << <1, 1>>> (self);
    // check for CUDA errors here
  }
};

#endif

} // namespace dynd