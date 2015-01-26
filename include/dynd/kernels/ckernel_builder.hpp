//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <new>
#include <algorithm>

#include <dynd/config.hpp>
#include <dynd/kernels/ckernel_prefix.hpp>
#include <dynd/types/type_id.hpp>

namespace dynd {

namespace kernels {
  /**
   * Increments a ``ckb_offset`` variable (offset into a ckernel_builder)
   * by the provided increment. The increment needs to be aligned to 8 bytes,
   * so padding may be added.
   */
  inline void inc_ckb_offset(intptr_t &inout_ckb_offset, size_t inc)
  {
    inout_ckb_offset +=
        static_cast<intptr_t>(ckernel_prefix::align_offset(inc));
  }

  template <class T>
  inline void inc_ckb_offset(intptr_t &inout_ckb_offset)
  {
    inc_ckb_offset(inout_ckb_offset, sizeof(T));
  }
} // namespace kernels

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
   *       that you're a leaf kernel, use ensure_capacity_leaf
   *       instead.
   */
  void ensure_capacity(intptr_t requested_capacity)
  {
    ensure_capacity_leaf(requested_capacity + sizeof(ckernel_prefix));
  }

  /**
   * This function ensures that the ckernel's data
   * is at least the required number of bytes. It
   * should only be called during the construction phase
   * of the kernel when constructing a leaf kernel.
   */
  void ensure_capacity_leaf(intptr_t requested_capacity)
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
    kernels::inc_ckb_offset<T>(inout_ckb_offset);
    ensure_capacity(inout_ckb_offset);
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
    kernels::inc_ckb_offset<T>(inout_ckb_offset);
    ensure_capacity_leaf(inout_ckb_offset);
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

  friend int ckernel_builder_ensure_capacity_leaf(void *ckb,
                                                  intptr_t requested_capacity);
};

#ifdef __CUDACC__

template <typename self_type, typename... A>
__global__ void cuda_device_init(ckernel_prefix *rawself,
                                 kernel_request_t kernreq, A... args)
{
  self_type::init(rawself, kernreq, args...);
}

__global__ void cuda_device_destroy(ckernel_prefix *self);

void throw_if_not_cuda_success(cudaError_t);

template <>
class ckernel_builder<kernel_request_cuda_device>
    : public base_ckernel_builder<ckernel_builder<kernel_request_cuda_device>> {
  static void *pool;
  static size_t pool_size;

public:
  void init()
  {
    m_data = reinterpret_cast<char *>(alloc(16 * 8));
    m_capacity = 16 * 8;
    set(m_data, 0, 16 * 8);
  }

  void *alloc(size_t size)
  {
    if (pool == NULL) {
      throw_if_not_cuda_success(cudaMalloc(&pool, size));
      pool_size = size;
    }
    if (pool_size == size) {
      return pool;
    }

    void *ptr;
    throw_if_not_cuda_success(cudaMalloc(&ptr, size));
    return ptr;
  }

  void *realloc(void *old_ptr, size_t old_size, size_t new_size)
  {
    void *new_ptr = alloc(new_size);
    copy(new_ptr, old_ptr, old_size);
    free(old_ptr);
    return new_ptr;
  }

  void free(void *ptr)
  {
    if (ptr == pool) {
      return;
    }
    throw_if_not_cuda_success(cudaFree(ptr));
  }

  void *copy(void *dst, const void *src, size_t size)
  {
    throw_if_not_cuda_success(
        cudaMemcpy(dst, src, size, cudaMemcpyDeviceToDevice));
    return dst;
  }

  void *set(void *dst, int value, size_t size)
  {
    throw_if_not_cuda_success(cudaMemset(dst, value, size));
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

/**
 * C API function for constructing a ckernel_builder object
 * in place. The `ckb` pointer must point to memory which
 * has sizeof(ckernel_builder) bytes (== 128 + 2 * sizeof(void *)),
 * and is aligned appropriately to 8 bytes.
 *
 * After a ckernel_builder instance is initialized this way,
 * all the other ckernel_builder functions can be used on it,
 * and when it is no longer needed, it must be destructed
 * by calling `ckernel_builder_destruct`.
 *
 * \param ckb  Pointer to the ckernel_builder instance. Must have
 *             128 + 2 * sizeof(void *), and alignment 8.
 */
inline void ckernel_builder_construct(void *ckb)
{
  // Use the placement new operator to initialize in-place
  new (ckb) ckernel_builder<kernel_request_host>();
}

/**
 * C API function for destroying a valid ckernel_builder object
 * in place. The `ckb` pointer must point to memory which
 * was previously initialized with `ckernel_buidler_construct`
 *
 * \param ckb  Pointer to the ckernel_builder instance.
 */
inline void ckernel_builder_destruct(void *ckb)
{
  // Call the destructor
  ckernel_builder<kernel_request_host> *ckb_ptr =
      reinterpret_cast<ckernel_builder<kernel_request_host> *>(ckb);
  ckb_ptr->~ckernel_builder<kernel_request_host>();
}

/**
 * C API function for resetting a valid ckernel_builder object
 * to an uninitialized state.
 *
 * \param ckb  Pointer to the ckernel_builder instance.
 */
inline void ckernel_builder_reset(void *ckb)
{
  ckernel_builder<kernel_request_host> *ckb_ptr =
      reinterpret_cast<ckernel_builder<kernel_request_host> *>(ckb);
  ckb_ptr->reset();
}

/**
 * C API function for ensuring that the kernel's data
 * is at least the required number of bytes. It
 * should only be called during the construction phase
 * of the kernel when constructing a leaf kernel.
 *
 * If the created kernel has a child kernel, use
 * the function `ckernel_builder_ensure_capacity` instead.
 *
 * \param ckb  Pointer to the ckernel_builder instance.
 * \param requested_capacity  The number of bytes required by the ckernel.
 *
 * \returns  0 on success, -1 on memory allocation failure.
 */
inline int ckernel_builder_ensure_capacity_leaf(void *ckb,
                                                intptr_t requested_capacity)
{
  ckernel_builder<kernel_request_host> *ckb_ptr =
      reinterpret_cast<ckernel_builder<kernel_request_host> *>(ckb);
  if (ckb_ptr->m_capacity < requested_capacity) {
    // Grow by a factor of 1.5
    // https://github.com/facebook/folly/blob/master/folly/docs/FBVector.md
    intptr_t grown_capacity = ckb_ptr->m_capacity * 3 / 2;
    if (requested_capacity < grown_capacity) {
      requested_capacity = grown_capacity;
    }
    char *new_data;
    if (ckb_ptr->using_static_data()) {
      // If we were previously using the static data, do a malloc
      new_data = reinterpret_cast<char *>(malloc(requested_capacity));
      // If the allocation succeeded, copy the old data as the realloc would
      if (new_data != NULL) {
        memcpy(new_data, ckb_ptr->m_data, ckb_ptr->m_capacity);
      }
    } else {
      // Otherwise do a realloc
      new_data = reinterpret_cast<char *>(
          realloc(ckb_ptr->m_data, requested_capacity));
    }
    if (new_data == NULL) {
      ckb_ptr->destroy();
      ckb_ptr->m_data = NULL;
      return -1;
    }
    // Zero out the newly allocated capacity
    memset(reinterpret_cast<char *>(new_data) + ckb_ptr->m_capacity, 0,
           requested_capacity - ckb_ptr->m_capacity);
    ckb_ptr->m_data = new_data;
    ckb_ptr->m_capacity = requested_capacity;
  }
  return 0;
}

/**
 * C API function for ensuring that the kernel's data
 * is at least the required number of bytes. It
 * should only be called during the construction phase
 * of the kernel when constructing a leaf kernel.
 *
 * This function allocates the requested capacity, plus enough
 * space for an empty child kernel to ensure safe destruction
 * during error handling. If a leaf kernel is being constructed,
 * use `ckernel_builder_ensure_capacity_leaf` instead.
 *
 * \param ckb  Pointer to the ckernel_builder instance.
 * \param requested_capacity  The number of bytes required by the ckernel.
 *
 * \returns  0 on success, -1 on memory allocation failure.
 */
inline int ckernel_builder_ensure_capacity(void *ckb,
                                           intptr_t requested_capacity)
{
  return ckernel_builder_ensure_capacity_leaf(ckb, requested_capacity +
                                                       sizeof(ckernel_prefix));
}

namespace kernels {
  /**
   * Some common shared implementation details of a CRTP
   * (curiously recurring template pattern) base class to help
   * create ckernels.
   */
  template <class CKT, kernel_request_t kernreq>
  struct general_ck;

#define GENERAL_CK(KERNREQ, ...)                                               \
  template <typename CKT>                                                      \
  struct general_ck<CKT, KERNREQ> {                                            \
    typedef CKT self_type;                                                     \
                                                                               \
    ckernel_prefix base;                                                       \
                                                                               \
    DYND_CUDA_HOST_DEVICE static self_type *get_self(ckernel_prefix *rawself)  \
    {                                                                          \
      return reinterpret_cast<self_type *>(rawself);                           \
    }                                                                          \
                                                                               \
    DYND_CUDA_HOST_DEVICE static const self_type *                             \
    get_self(const ckernel_prefix *rawself)                                    \
    {                                                                          \
      return reinterpret_cast<const self_type *>(rawself);                     \
    }                                                                          \
                                                                               \
    template <typename CKBT>                                                   \
    static self_type *get_self(CKBT *ckb, intptr_t ckb_offset)                 \
    {                                                                          \
      return ckb->template get_at<self_type>(ckb_offset);                      \
    }                                                                          \
                                                                               \
    /**                                                                        \
     * Creates the ckernel, and increments ``inckb_offset``                    \
     * to the position after it.                                               \
     */                                                                        \
    template <typename CKBT, typename... A>                                    \
    static self_type *create(CKBT *ckb, kernel_request_t kernreq,              \
                             intptr_t &inout_ckb_offset, A &&... args)         \
    {                                                                          \
      intptr_t ckb_offset = inout_ckb_offset;                                  \
      kernels::inc_ckb_offset<self_type>(inout_ckb_offset);                    \
      ckb->ensure_capacity(inout_ckb_offset);                                  \
      ckernel_prefix *rawself =                                                \
          ckb->template get_at<ckernel_prefix>(ckb_offset);                    \
      return ckb->template init<self_type>(rawself, kernreq,                   \
                                           std::forward<A>(args)...);          \
    }                                                                          \
                                                                               \
    template <typename... A>                                                   \
    static self_type *create(void *ckb, kernel_request_t kernreq,              \
                             intptr_t &inout_ckb_offset, A &&... args);        \
                                                                               \
    /**                                                                        \
     * Creates the ckernel, and increments ``inckb_offset``                    \
     * to the position after it.                                               \
     */                                                                        \
    template <typename CKBT, typename... A>                                    \
    static self_type *create_leaf(CKBT *ckb, kernel_request_t kernreq,         \
                                  intptr_t &inout_ckb_offset, A &&... args)    \
    {                                                                          \
      intptr_t ckb_offset = inout_ckb_offset;                                  \
      kernels::inc_ckb_offset<self_type>(inout_ckb_offset);                    \
      ckb->ensure_capacity_leaf(inout_ckb_offset);                             \
      ckernel_prefix *rawself =                                                \
          ckb->template get_at<ckernel_prefix>(ckb_offset);                    \
      return ckb->template init<self_type>(rawself, kernreq,                   \
                                           std::forward<A>(args)...);          \
    }                                                                          \
                                                                               \
    template <typename... A>                                                   \
    static self_type *create_leaf(void *ckb, kernel_request_t kernreq,         \
                                  intptr_t &inout_ckb_offset, A &&... args);   \
                                                                               \
    /**                                                                        \
     * Initializes an instance of this ckernel in-place according to the       \
     * kernel request. This calls the constructor in-place, and initializes    \
     * the base function and destructor.                                       \
     */                                                                        \
    template <typename... A>                                                   \
    __VA_ARGS__ static self_type *init(ckernel_prefix *rawself,                \
                                       kernel_request_t kernreq, A &&... args) \
    {                                                                          \
      /* Alignment requirement of the type. */                                 \
      DYND_STATIC_ASSERT((size_t)scalar_align_of<self_type>::value <=          \
                             (size_t)scalar_align_of<uint64_t>::value,         \
                         "ckernel types require alignment <= 64 bits");        \
                                                                               \
      /* Call the constructor in-place. */                                     \
      self_type *self = new (rawself) self_type(args...);                      \
      /* Double check that the C++ struct layout is as we expect. */           \
      if (self != get_self(rawself)) {                                         \
        DYND_HOST_THROW(std::runtime_error,                                    \
                        "internal ckernel error: struct layout is not valid"); \
      }                                                                        \
      self->base.destructor = &self_type::destruct;                            \
      /* A child class must implement this to fill in self->base.function. */  \
      self->init_kernfunc(kernreq);                                            \
      return self;                                                             \
    }                                                                          \
                                                                               \
    /**                                                                        \
     * The ckernel destructor function, which is placed in                     \
     * base.destructor.                                                        \
     */                                                                        \
    __VA_ARGS__ static void destruct(ckernel_prefix *rawself)                  \
    {                                                                          \
      self_type *self = get_self(rawself);                                     \
      /* If there are any child kernels, a child class must implement this to  \
       * destroy them. */                                                      \
      self->destruct_children();                                               \
      self->~self_type();                                                      \
    }                                                                          \
                                                                               \
    /**                                                                        \
     * Default implementation of destruct_children does nothing.               \
     */                                                                        \
    __VA_ARGS__ void destruct_children() {}                                    \
                                                                               \
    /**                                                                        \
     * Returns the child ckernel immediately following this one.               \
     */                                                                        \
    __VA_ARGS__ ckernel_prefix *get_child_ckernel()                            \
    {                                                                          \
      return get_child_ckernel(sizeof(self_type));                             \
    }                                                                          \
                                                                               \
    /**                                                                        \
     * Returns the child ckernel at the specified offset.                      \
     */                                                                        \
    __VA_ARGS__ ckernel_prefix *get_child_ckernel(intptr_t offset)             \
    {                                                                          \
      return base.get_child_ckernel(ckernel_prefix::align_offset(offset));     \
    }                                                                          \
  };

  GENERAL_CK(kernel_request_host);

  template <typename CKT>
  template <typename... A>
  typename general_ck<CKT, kernel_request_host>::self_type *
  general_ck<CKT, kernel_request_host>::create(void *ckb,
                                               kernel_request_t kernreq,
                                               intptr_t &inout_ckb_offset,
                                               A &&... args)
  {
    switch (kernreq & kernel_request_memory) {
    case kernel_request_host:
      return self_type::create(
          reinterpret_cast<ckernel_builder<kernel_request_host> *>(ckb),
          kernreq, inout_ckb_offset, std::forward<A>(args)...);
    default:
      throw std::invalid_argument("unrecognized ckernel request");
    }
  }

  template <typename CKT>
  template <typename... A>
  typename general_ck<CKT, kernel_request_host>::self_type *
  general_ck<CKT, kernel_request_host>::create_leaf(void *ckb,
                                                    kernel_request_t kernreq,
                                                    intptr_t &inout_ckb_offset,
                                                    A &&... args)
  {
    switch (kernreq & kernel_request_memory) {
    case kernel_request_host:
      return self_type::create_leaf(
          reinterpret_cast<ckernel_builder<kernel_request_host> *>(ckb),
          kernreq, inout_ckb_offset, std::forward<A>(args)...);
    default:
      throw std::invalid_argument("unrecognized ckernel request");
    }
  }

#ifdef __CUDACC__

  GENERAL_CK(kernel_request_cuda_device, __device__);

  template <typename CKT>
  template <typename... A>
  typename general_ck<CKT, kernel_request_cuda_device>::self_type *
  general_ck<CKT, kernel_request_cuda_device>::create(
      void *ckb, kernel_request_t kernreq, intptr_t &inout_ckb_offset,
      A &&... args)
  {
    switch (kernreq & kernel_request_memory) {
    case kernel_request_cuda_device:
      return self_type::create(
          reinterpret_cast<ckernel_builder<kernel_request_cuda_device> *>(ckb),
          kernreq & ~kernel_request_cuda_device, inout_ckb_offset,
          std::forward<A>(args)...);
    default:
      throw std::invalid_argument("unrecognized ckernel request");
    }
  }

  template <typename CKT>
  template <typename... A>
  typename general_ck<CKT, kernel_request_cuda_device>::self_type *
  general_ck<CKT, kernel_request_cuda_device>::create_leaf(
      void *ckb, kernel_request_t kernreq, intptr_t &inout_ckb_offset,
      A &&... args)
  {
    switch (kernreq & kernel_request_memory) {
    case kernel_request_cuda_device:
      return self_type::create_leaf(
          reinterpret_cast<ckernel_builder<kernel_request_cuda_device> *>(ckb),
          kernreq & ~kernel_request_cuda_device, inout_ckb_offset,
          std::forward<A>(args)...);
    default:
      throw std::invalid_argument("unrecognized ckernel request");
    }
  }

#endif

#ifdef DYND_CUDA

  GENERAL_CK(kernel_request_cuda_host_device, DYND_CUDA_HOST_DEVICE);

  template <typename CKT>
  template <typename... A>
  typename general_ck<CKT, kernel_request_cuda_host_device>::self_type *
  general_ck<CKT, kernel_request_cuda_host_device>::create(
      void *ckb, kernel_request_t kernreq, intptr_t &inout_ckb_offset,
      A &&... args)
  {
    switch (kernreq & kernel_request_memory) {
    case kernel_request_host:
      return self_type::create(
          reinterpret_cast<ckernel_builder<kernel_request_host> *>(ckb),
          kernreq, inout_ckb_offset, std::forward<A>(args)...);
#ifdef __CUDACC__
    case kernel_request_cuda_device:
      return self_type::create(
          reinterpret_cast<ckernel_builder<kernel_request_cuda_device> *>(ckb),
          kernreq & ~kernel_request_cuda_device, inout_ckb_offset,
          std::forward<A>(args)...);
#endif
    default:
      throw std::invalid_argument("unrecognized ckernel request");
    }
  }

  template <typename CKT>
  template <typename... A>
  typename general_ck<CKT, kernel_request_cuda_host_device>::self_type *
  general_ck<CKT, kernel_request_cuda_host_device>::create_leaf(
      void *ckb, kernel_request_t kernreq, intptr_t &inout_ckb_offset,
      A &&... args)
  {
    switch (kernreq & kernel_request_memory) {
    case kernel_request_host:
      return self_type::create_leaf(
          reinterpret_cast<ckernel_builder<kernel_request_host> *>(ckb),
          kernreq, inout_ckb_offset, std::forward<A>(args)...);
#ifdef __CUDACC__
    case kernel_request_cuda_device:
      return self_type::create_leaf(
          reinterpret_cast<ckernel_builder<kernel_request_cuda_device> *>(ckb),
          kernreq & ~kernel_request_cuda_device, inout_ckb_offset,
          std::forward<A>(args)...);
#endif
    default:
      throw std::invalid_argument("unrecognized ckernel request");
    }
  }

#endif

#undef GENERAL_CK

  typedef void *(*create_t)(void *, kernel_request_t, intptr_t &);

  template <typename CKT>
  void *create(void *ckb, kernel_request_t kernreq, intptr_t &inout_ckb_offset)
  {
    return CKT::create(ckb, kernreq, inout_ckb_offset);
  }

} // namespace kernels
} // namespace dynd