//
// Copyright (C) 2011-13 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//
// DEPRECATED

#ifndef _DYND__KERNEL_INSTANCE_HPP_
#define _DYND__KERNEL_INSTANCE_HPP_

#include <dynd/auxiliary_data.hpp>

namespace dynd {

/**
 * Data which remains static across multiple binary operation calls.
 */
struct binary_kernel_static_data {
    auxiliary_data auxdata;
    const char *dst_metadata, *src0_metadata, *src1_metadata;

    binary_kernel_static_data()
        : auxdata(), dst_metadata(), src0_metadata(), src1_metadata()
    {}
    binary_kernel_static_data(const binary_kernel_static_data& rhs)
        : auxdata(), dst_metadata(rhs.dst_metadata), src0_metadata(rhs.src0_metadata), src1_metadata(rhs.src1_metadata)
    {
        auxdata.clone_from(rhs.auxdata);
    }
    void clone_from(const binary_kernel_static_data& rhs)
    {
        auxdata.clone_from(rhs.auxdata);
        dst_metadata = rhs.dst_metadata;
        src0_metadata = rhs.src0_metadata;
        src1_metadata = rhs.src1_metadata;
    }
    void borrow_from(const binary_kernel_static_data& rhs)
    {
        auxdata.borrow_from(rhs.auxdata);
        dst_metadata = rhs.dst_metadata;
        src0_metadata = rhs.src0_metadata;
        src1_metadata = rhs.src1_metadata;
    }
    void swap(binary_kernel_static_data& rhs)
    {
        auxdata.swap(rhs.auxdata);
        std::swap(dst_metadata, rhs.dst_metadata);
        std::swap(src0_metadata, rhs.src0_metadata);
        std::swap(src1_metadata, rhs.src1_metadata);
    }
};


/** Typedef for a binary operation on a single element */
typedef void (*binary_single_operation_t)(char *dst, const char *src0, const char *src1, binary_kernel_static_data *extra);
/** Typedef for a binary operation on a strided segment of elements */
typedef void (*binary_strided_operation_t)(char *dst, intptr_t dst_stride,
                const char *src0, intptr_t src0_stride,
                const char *src1, intptr_t src1_stride,
                size_t count, binary_kernel_static_data *extra);


/**
 * A structure containing a pair of binary operation pointers, for single and strided elements.
 * It is permitted for the 'contig' specialization to be NULL, in which case any code using this
 * must call the 'single' version repeatedly.
 */
struct binary_operation_pair_t {
    binary_single_operation_t single;
    binary_strided_operation_t strided;

    typedef binary_kernel_static_data extra_type;

    binary_operation_pair_t()
        : single(), strided()
    {}
    binary_operation_pair_t(binary_single_operation_t single_, binary_strided_operation_t strided_)
        : single(single_), strided(strided_)
    {}
};

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
        : kernel(), extra()
    {
    }
    // Copying a kernel_instance clones the auxiliary data
    kernel_instance(const kernel_instance& rhs)
        : kernel(rhs.kernel), extra(rhs.extra)
    {
    }

    void swap(kernel_instance& rhs) {
        std::swap(kernel, rhs.kernel);
        extra.swap(rhs.extra);
    }

    void copy_from(const kernel_instance& rhs) {
        kernel = rhs.kernel;
        extra.clone_from(rhs.extra);
    }

    void borrow_from(const kernel_instance& rhs) {
        kernel = rhs.kernel;
        extra.borrow_from(rhs.extra);
    }

    /** The kernel functions */
    FT kernel;
    /** The structure that contains extra data, including auxiliary data and the operands' metadata */
    typename FT::extra_type extra;
};

} // namespace dynd;

#endif // _DYND__KERNEL_INSTANCE_HPP_
