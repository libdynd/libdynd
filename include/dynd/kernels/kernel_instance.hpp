//
// Copyright (C) 2011-12, Dynamic NDArray Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DYND__KERNEL_INSTANCE_HPP_
#define _DYND__KERNEL_INSTANCE_HPP_

#include <dynd/auxiliary_data.hpp>

namespace dynd {

/**
 * Data which remains static across multiple unary operation calls.
 */
struct unary_kernel_static_data {
    AuxDataBase *auxdata;
    const char *dst_metadata, *src_metadata;

    unary_kernel_static_data()
        : auxdata(), dst_metadata(), src_metadata()
    {}
    unary_kernel_static_data(AuxDataBase *ad, const char *dm, const char *sm)
        : auxdata(ad), dst_metadata(dm), src_metadata(sm)
    {}
};

/** Typedef for a unary operation on a single element */
typedef void (*unary_single_operation_t)(char *dst, const char *src, unary_kernel_static_data *extra);
/** Typedef for a unary operation on a contiguous segment of elements */
typedef void (*unary_contig_operation_t)(char *dst, const char *src, size_t count, unary_kernel_static_data *extra);

/**
 * A structure containing a pair of unary operation pointers, for single and contiguous elements.
 * It is permitted for the 'contig' specialization to be NULL, in which case any code using this
 * must call the 'single' version repeatedly.
 */
struct unary_operation_pair_t {
    unary_single_operation_t single;
    unary_contig_operation_t contig;

    unary_operation_pair_t()
        : single(), contig()
    {}
    unary_operation_pair_t(unary_single_operation_t s, unary_contig_operation_t c)
        : single(s), contig(c)
    {}
};


typedef void (*nullary_operation_t)(char *dst, intptr_t dst_stride,
                        intptr_t count, const AuxDataBase *auxdata);

typedef void (*binary_operation_t)(char *dst, intptr_t dst_stride,
                        const char *src0, intptr_t src0_stride,
                        const char *src1, intptr_t src1_stride,
                        intptr_t count, const AuxDataBase *auxdata);

typedef bool (*single_compare_operation_t)(const char *src0, const char *src1,
                        const AuxDataBase *auxdata);

/**
 * This class holds an instance of a kernel function, with its
 * associated auxiliary data. The object is non-copyable, just
 * like the auxiliary_data object, to avoid inefficient copies.
 */
template<typename FT>
class kernel_instance {
    kernel_instance& operator=(const kernel_instance&);
public:
    kernel_instance()
        : kernel()
    {
    }
    // Copying a kernel_instance clones the auxiliary data
    kernel_instance(const kernel_instance& rhs)
        : kernel(rhs.kernel)
    {
        auxdata.clone_from(rhs.auxdata);
    }

    void swap(kernel_instance& rhs) {
        std::swap(kernel, rhs.kernel);
        auxdata.swap(rhs.auxdata);
    }

    void copy_from(const kernel_instance& rhs) {
        kernel = rhs.kernel;
        auxdata.clone_from(rhs.auxdata);
    }

    void borrow_from(const kernel_instance& rhs) {
        kernel = rhs.kernel;
        auxdata.borrow_from(rhs.auxdata);
    }

    FT kernel;
    auxiliary_data auxdata;
};

} // namespace dynd;

#endif // _DYND__KERNEL_INSTANCE_HPP_
