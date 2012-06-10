//
// Copyright (C) 2011 Mark Wiebe (mwwiebe@gmail.com)
// All rights reserved.
//
// This is unreleased proprietary software.
//

#include <iostream> // FOR DEBUG
#include <typeinfo> // FOR DEBUG

#include <sstream>
#include <stdexcept>
#include <cstring>
#include <limits>

#include <dnd/dtype_assign.hpp>
#include <dnd/dtypes/conversion_dtype.hpp>
#include <dnd/kernels/assignment_kernels.hpp>

#ifdef __GNUC__
// The -Weffc++ flag warns about derived classes not having a virtual destructor.
// Here, this is explicitly done, because we are only using derived classes
// to inherit a static function, they are never instantiated.
//
// NOTE: The documentation says this is only for g++ 4.6.0 and up.
#pragma GCC diagnostic ignored "-Weffc++"
#endif


using namespace std;
using namespace dnd;

std::ostream& dnd::operator<<(ostream& o, assign_error_mode errmode)
{
    switch (errmode) {
    case assign_error_none:
        o << "none";
        break;
    case assign_error_overflow:
        o << "overflow";
        break;
    case assign_error_fractional:
        o << "fractional";
        break;
    case assign_error_inexact:
        o << "inexact";
        break;
    default:
        o << "invalid error mode(" << (int)errmode << ")";
        break;
    }

    return o;
}

// Returns true if the destination dtype can represent *all* the values
// of the source dtype, false otherwise. This is used, for example,
// to skip any overflow checks when doing value assignments between differing
// types.
bool dnd::is_lossless_assignment(const dtype& dst_dt, const dtype& src_dt)
{
    const extended_dtype *dst_ext, *src_ext;

    dst_ext = dst_dt.extended();
    src_ext = src_dt.extended();

    if (dst_ext == NULL && src_ext == NULL) {
        switch (src_dt.kind()) {
            case pattern_kind: // TODO: raise an error?
                return true;
            case bool_kind:
                switch (dst_dt.kind()) {
                    case bool_kind:
                    case int_kind:
                    case uint_kind:
                    case real_kind:
                    case complex_kind:
                        return true;
                    case string_kind:
                        return dst_dt.itemsize() > 0;
                    default:
                        break;
                }
                break;
            case int_kind:
                switch (dst_dt.kind()) {
                    case bool_kind:
                        return false;
                    case int_kind:
                        return dst_dt.itemsize() >= src_dt.itemsize();
                    case uint_kind:
                        return false;
                    case real_kind:
                        return dst_dt.itemsize() > src_dt.itemsize();
                    case complex_kind:
                        return dst_dt.itemsize() > 2 * src_dt.itemsize();
                    case string_kind:
                        // Conservative value for 64-bit, could
                        // check speciifically based on the type_id.
                        return dst_dt.itemsize() >= 21;
                    default:
                        break;
                }
                break;
            case uint_kind:
                switch (dst_dt.kind()) {
                    case bool_kind:
                        return false;
                    case int_kind:
                        return dst_dt.itemsize() > src_dt.itemsize();
                    case uint_kind:
                        return dst_dt.itemsize() >= src_dt.itemsize();
                    case real_kind:
                        return dst_dt.itemsize() > src_dt.itemsize();
                    case complex_kind:
                        return dst_dt.itemsize() > 2 * src_dt.itemsize();
                    case string_kind:
                        // Conservative value for 64-bit, could
                        // check speciifically based on the type_id.
                        return dst_dt.itemsize() >= 21;
                    default:
                        break;
                }
                break;
            case real_kind:
                switch (dst_dt.kind()) {
                    case bool_kind:
                    case int_kind:
                    case uint_kind:
                        return false;
                    case real_kind:
                        return dst_dt.itemsize() >= src_dt.itemsize();
                    case complex_kind:
                        return dst_dt.itemsize() >= 2 * src_dt.itemsize();
                    case string_kind:
                        return dst_dt.itemsize() >= 32;
                    default:
                        break;
                }
            case complex_kind:
                switch (dst_dt.kind()) {
                    case bool_kind:
                    case int_kind:
                    case uint_kind:
                    case real_kind:
                        return false;
                    case complex_kind:
                        return dst_dt.itemsize() >= src_dt.itemsize();
                    case string_kind:
                        return dst_dt.itemsize() >= 64;
                    default:
                        break;
                }
            case string_kind:
                switch (dst_dt.kind()) {
                    case bool_kind:
                    case int_kind:
                    case uint_kind:
                    case real_kind:
                    case complex_kind:
                        return false;
                    case string_kind:
                        return src_dt.type_id() == dst_dt.type_id() &&
                                dst_dt.itemsize() >= src_dt.itemsize();
                    default:
                        break;
                }
            default:
                break;
        }

        throw std::runtime_error("unhandled built-in case in is_lossless_assignmently");
    }

    // Use the available extended_dtype to check the casting
    if (src_ext != NULL) {
        return src_ext->is_lossless_assignment(dst_dt, src_dt);
    }
    else {
        return dst_ext->is_lossless_assignment(dst_dt, src_dt);
    }
}


void dnd::dtype_assign(const dtype& dst_dt, char *dst, const dtype& src_dt, const char *src, assign_error_mode errmode)
{
    if (dst_dt.extended() == NULL && src_dt.extended() == NULL) {
        // Try to use the simple single-value assignment for built-in types
        assignment_function_t asn = get_builtin_dtype_assignment_function(dst_dt.type_id(), src_dt.type_id(), errmode);
        if (asn != NULL) {
            asn(dst, src);
            return;
        }

        stringstream ss;
        ss << "assignment from " << src_dt << " to " << dst_dt << " isn't yet supported";
        throw std::runtime_error(ss.str());
    } else {
        // Fall back to the strided assignment functions for the extended dtypes
        kernel_instance<unary_operation_t> op;
        get_dtype_assignment_kernel(dst_dt, 0, src_dt, 0, errmode, op);
        op.kernel(dst, 0, src, 0, 1, op.auxdata);
        return;
    }
}

/*
// A multiple unaligned byteswap assignment function which uses one of the single assignment functions as proxy
namespace {
    class multiple_byteswap_unaligned_auxiliary_data : public auxiliary_data {
    public:
        assign_function_t assign;
        byteswap_operation_t src_byteswap, dst_byteswap;
        int dst_itemsize, src_itemsize;
        multiple_byteswap_unaligned_auxiliary_data()
            : assign(NULL), src_byteswap(NULL), dst_byteswap(NULL),
              dst_itemsize(0), src_itemsize(0) {
        }

        virtual ~multiple_byteswap_unaligned_auxiliary_data() {
        }
    };
}
static void assign_multiple_byteswap_unaligned(char *dst, intptr_t dst_stride,
                                    const char *src, intptr_t src_stride,
                                    intptr_t count, const auxiliary_data *data)
{
    const multiple_byteswap_unaligned_auxiliary_data * mgdata =
                            static_cast<const multiple_byteswap_unaligned_auxiliary_data *>(data);

    char *dst_cached = reinterpret_cast<char *>(dst);
    const char *src_cached = reinterpret_cast<const char *>(src);

    byteswap_operation_t src_byteswap = mgdata->src_byteswap, dst_byteswap = mgdata->dst_byteswap;
    int dst_itemsize = mgdata->dst_itemsize, src_itemsize = mgdata->src_itemsize;
    // TODO: Probably want to relax the assumption of at most 8 bytes
    int64_t d;
    int64_t s;

    assign_function_t asn = mgdata->assign;

    for (intptr_t i = 0; i < count; ++i) {
        src_byteswap(&s, src_cached, src_itemsize);
        asn(&d, &s);
        dst_byteswap(dst_cached, &d, dst_itemsize);
        dst_cached += dst_stride;
        src_cached += src_stride;
    }
}

// A multiple unaligned assignment function which uses one of the single assignment functions as proxy
namespace {
    struct multiple_unaligned_auxiliary_data {
        assign_function_t assign;
        int dst_itemsize, src_itemsize;
    };
}
static void assign_multiple_unaligned(char *dst, intptr_t dst_stride, const char *src, intptr_t src_stride,
                                    intptr_t count, const AuxDataBase *auxdata)
{
    const multiple_unaligned_auxiliary_data &mgdata = get_auxiliary_data<multiple_unaligned_auxiliary_data>(auxdata);

    char *dst_cached = reinterpret_cast<char *>(dst);
    const char *src_cached = reinterpret_cast<const char *>(src);

    int dst_itemsize = mgdata.dst_itemsize, src_itemsize = mgdata.src_itemsize;
    // TODO: Probably want to relax the assumption of at most 8 bytes
    int64_t d;
    int64_t s;

    assign_function_t asn = mgdata.assign;

    for (intptr_t i = 0; i < count; ++i) {
        memcpy(&s, src_cached, src_itemsize);
        asn(&d, &s);
        memcpy(dst_cached, &d, dst_itemsize);
        dst_cached += dst_stride;
        src_cached += src_stride;
    }
}
*/


void dnd::dtype_strided_assign(const dtype& dst_dt, char *dst, intptr_t dst_stride,
                            const dtype& src_dt, const char *src, intptr_t src_stride,
                            intptr_t count, assign_error_mode errmode)
{
    kernel_instance<unary_operation_t> op;
    get_dtype_assignment_kernel(dst_dt, dst_stride,
                                src_dt, src_stride,
                                errmode, op);
    op.kernel(dst, dst_stride, src, src_stride, count, op.auxdata);
}

// Fixed and unknown size contiguous copy assignment functions
template<int N>
static void contig_fixedsize_copy_assign(char *dst, intptr_t, const char *src, intptr_t,
                            intptr_t count, const AuxDataBase *) {
    memcpy(dst, src, N * count);
}
namespace {
    template<class T>
    struct fixed_size_copy_assign_type {
        static void assign(char *dst, intptr_t dst_stride, const char *src, intptr_t src_stride,
                            intptr_t count, const AuxDataBase *) {
            T *dst_cached = reinterpret_cast<T *>(dst);
            const T *src_cached = reinterpret_cast<const T *>(src);
            dst_stride /= sizeof(T);
            src_stride /= sizeof(T);

            for (intptr_t i = 0; i < count; ++i) {
                *dst_cached = *src_cached;
                
                dst_cached += dst_stride;
                src_cached += src_stride;
            }
        }
    };

    template<int N>
    struct fixed_size_copy_assign;
    template<>
    struct fixed_size_copy_assign<1> : public fixed_size_copy_assign_type<char> {};
    template<>
    struct fixed_size_copy_assign<2> : public fixed_size_copy_assign_type<int16_t> {};
    template<>
    struct fixed_size_copy_assign<4> : public fixed_size_copy_assign_type<int32_t> {};
    template<>
    struct fixed_size_copy_assign<8> : public fixed_size_copy_assign_type<int64_t> {};

    template<class T>
    struct fixed_size_copy_zerostride_assign_type {
        static void assign(char *dst, intptr_t dst_stride, const char *src, intptr_t,
                            intptr_t count, const AuxDataBase *) {
            T *dst_cached = reinterpret_cast<T *>(dst);
            T s = *reinterpret_cast<const T *>(src);
            dst_stride /= sizeof(T);

            for (intptr_t i = 0; i < count; ++i) {
                *dst_cached = s;
                
                dst_cached += dst_stride;
            }
        }
    };

    template<int N>
    struct fixed_size_copy_zerostride_assign;
    template<>
    struct fixed_size_copy_zerostride_assign<1> : public fixed_size_copy_zerostride_assign_type<char> {};
    template<>
    struct fixed_size_copy_zerostride_assign<2> : public fixed_size_copy_zerostride_assign_type<int16_t> {};
    template<>
    struct fixed_size_copy_zerostride_assign<4> : public fixed_size_copy_zerostride_assign_type<int32_t> {};
    template<>
    struct fixed_size_copy_zerostride_assign<8> : public fixed_size_copy_zerostride_assign_type<int64_t> {};
}
static void contig_copy_assign(char *dst, intptr_t, const char *src, intptr_t,
                            intptr_t count, const AuxDataBase *auxdata)
{
    intptr_t itemsize = get_auxiliary_data<intptr_t>(auxdata);
    memcpy(dst, src, itemsize * count);
}
static void strided_copy_assign(char *dst, intptr_t dst_stride, const char *src, intptr_t src_stride,
                            intptr_t count, const AuxDataBase *auxdata)
{
    char *dst_cached = reinterpret_cast<char *>(dst);
    const char *src_cached = reinterpret_cast<const char *>(src);
    intptr_t itemsize = get_auxiliary_data<intptr_t>(auxdata);

    for (intptr_t i = 0; i < count; ++i) {
        memcpy(dst_cached, src_cached, itemsize);
        dst_cached += dst_stride;
        src_cached += src_stride;
    }
}
static void fixed_size_copy_contig_zerostride_assign_memset(char *dst, intptr_t, const char *src, intptr_t,
                            intptr_t count, const AuxDataBase *)
{
    char s = *reinterpret_cast<const char *>(src);
    memset(dst, s, count);
}
static void strided_copy_zerostride_assign(char *dst, intptr_t dst_stride, const char *src, intptr_t,
                            intptr_t count, const AuxDataBase *auxdata)
{
    char *dst_cached = reinterpret_cast<char *>(dst);
    intptr_t itemsize = get_auxiliary_data<intptr_t>(auxdata);

    for (intptr_t i = 0; i < count; ++i) {
        memcpy(dst_cached, src, itemsize);
        dst_cached += dst_stride;
    }
}


void dnd::get_dtype_strided_assign_operation(
                    const dtype& dt,
                    intptr_t dst_fixedstride,
                    intptr_t src_fixedstride,
                    kernel_instance<unary_operation_t>& out_kernel)
{
    // Make sure there's no stray auxiliary data
    out_kernel.auxdata.free();

    //DEBUG_COUT << "get_dtype_strided_assign_operation (single dtype " << dt << ")\n";
    if (!dt.is_object_type()) {
        if (dst_fixedstride == (intptr_t)dt.itemsize() &&
                                    src_fixedstride == (intptr_t)dt.itemsize()) {
            // contig -> contig uses memcpy, works with unaligned data
            switch (dt.itemsize()) {
                case 1:
                    out_kernel.kernel = &contig_fixedsize_copy_assign<1>;
                    break;
                case 2:
                    out_kernel.kernel = &contig_fixedsize_copy_assign<2>;
                    break;
                case 4:
                    out_kernel.kernel = &contig_fixedsize_copy_assign<4>;
                    break;
                case 8:
                    out_kernel.kernel = &contig_fixedsize_copy_assign<8>;
                    break;
                case 16:
                    out_kernel.kernel = &contig_fixedsize_copy_assign<16>;
                    break;
                default:
                    out_kernel.kernel = &contig_copy_assign;
                    make_auxiliary_data<intptr_t>(out_kernel.auxdata, dt.itemsize());
                    break;
            }
        } else if (src_fixedstride == 0) {
            out_kernel.kernel = NULL;
            switch (dt.itemsize()) {
                case 1:
                    if (dst_fixedstride == 1) {
                        out_kernel.kernel = &fixed_size_copy_contig_zerostride_assign_memset;
                    } else {
                        out_kernel.kernel = &fixed_size_copy_zerostride_assign<1>::assign;
                    }
                    break;
                case 2:
                    out_kernel.kernel = &fixed_size_copy_zerostride_assign<2>::assign;
                    break;
                case 4:
                    out_kernel.kernel = &fixed_size_copy_zerostride_assign<4>::assign;
                    break;
                case 8:
                    out_kernel.kernel = &fixed_size_copy_zerostride_assign<8>::assign;
                    break;
            }

            if (out_kernel.kernel == NULL) {
                out_kernel.kernel = &strided_copy_zerostride_assign;
                make_auxiliary_data<intptr_t>(out_kernel.auxdata, dt.itemsize());
            }
        } else {
            out_kernel.kernel = NULL;
            switch (dt.itemsize()) {
                case 1:
                    out_kernel.kernel = &fixed_size_copy_assign<1>::assign;
                    break;
                case 2:
                    out_kernel.kernel = &fixed_size_copy_assign<2>::assign;
                    break;
                case 4:
                    out_kernel.kernel = &fixed_size_copy_assign<4>::assign;
                    break;
                case 8:
                    out_kernel.kernel = &fixed_size_copy_assign<8>::assign;
                    break;
            }

            if (out_kernel.kernel == NULL) {
                out_kernel.kernel = &strided_copy_assign;
                make_auxiliary_data<intptr_t>(out_kernel.auxdata, dt.itemsize());
            }
        }
    } else {
        throw std::runtime_error("cannot assign object dtypes yet");
    }
}
