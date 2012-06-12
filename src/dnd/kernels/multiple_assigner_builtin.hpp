//
// Copyright (C) 2012 Continuum Analytics
// All rights reserved.
//
#ifndef _DND__MULTIPLE_ASSIGNER_BUILTIN_HPP_
#define _DND__MULTIPLE_ASSIGNER_BUILTIN_HPP_

#include <dnd/diagnostics.hpp>
#include "single_assigner_builtin.hpp"

// Put it in an anonymous namespace
namespace dnd {

// Some specialized multiple assignment functions
template<class dst_type, class src_type, assign_error_mode errmode>
struct multiple_assigner_builtin {
    static void assign_anystride_anystride(char *dst, intptr_t dst_stride,
                                const char *src, intptr_t src_stride,
                                intptr_t count,
                                const AuxDataBase *)
    {
        DND_ASSERT_ALIGNED(dst, dst_stride, sizeof(dst_type),
                "src type: " << dnd::dtype(dnd::type_id_of<src_type>::value) << "  dst type id: " << dnd::dtype(dnd::type_id_of<dst_type>::value));
        DND_ASSERT_ALIGNED(src, src_stride, sizeof(src_type),
                "src type: " << dnd::dtype(dnd::type_id_of<src_type>::value) << "  dst type id: " << dnd::dtype(dnd::type_id_of<dst_type>::value));
        //DEBUG_COUT << "multiple_assigner::assign_noexcept (" << typeid(src_type).name() << " -> " << typeid(dst_type).name() << ")\n";
        const src_type *src_cached = reinterpret_cast<const src_type *>(src);
        dst_type *dst_cached = reinterpret_cast<dst_type *>(dst);
        src_stride /= sizeof(src_type);
        dst_stride /= sizeof(dst_type);

        for (intptr_t i = 0; i < count; ++i) {
            single_assigner_builtin<dst_type, src_type, errmode>::assign(dst_cached, src_cached);
            dst_cached += dst_stride;
            src_cached += src_stride;
        }
    }

    static void assign_anystride_zerostride(char *dst, intptr_t dst_stride,
                                const char *src, intptr_t,
                                intptr_t count,
                                const AuxDataBase *)
    {
        //std::cout << "doing cast" << " src type: " << dnd::dtype(dnd::type_id_of<src_type>::value) << "  dst type: " << dnd::dtype(dnd::type_id_of<dst_type>::value) << std::endl;
        //std::cout << "dst ptr " << (void *)dst << ", dst stride " << dst_stride << std::endl;

        DND_ASSERT_ALIGNED(dst, dst_stride, sizeof(dst_type),
                "src type: " << dnd::dtype(dnd::type_id_of<src_type>::value) << "  dst type: " << dnd::dtype(dnd::type_id_of<dst_type>::value));
        DND_ASSERT_ALIGNED(src, 0, sizeof(src_type),
                "src type: " << dnd::dtype(dnd::type_id_of<src_type>::value) << "  dst type id: " << dnd::dtype(dnd::type_id_of<dst_type>::value));
        //DEBUG_COUT << "multiple_assigner::assign_noexcept_anystride_zerostride (" << typeid(src_type).name() << " -> " << typeid(dst_type).name() << ")\n";
        dst_type src_value_cached;
        dst_type *dst_cached = reinterpret_cast<dst_type *>(dst);
        dst_stride /= sizeof(dst_type);

        single_assigner_builtin<dst_type, src_type, errmode>::assign(&src_value_cached, (const src_type *)src);

        for (intptr_t i = 0; i < count; ++i) {
            *dst_cached = src_value_cached;
            dst_cached += dst_stride;
        }
    }

    static void assign_contigstride_zerostride(char *dst, intptr_t,
                                const char *src, intptr_t,
                                intptr_t count,
                                const AuxDataBase *)
    {
        DND_ASSERT_ALIGNED(dst, sizeof(dst_type), sizeof(dst_type),
                "src type: " << dnd::dtype(dnd::type_id_of<src_type>::value) << "  dst type id: " << dnd::dtype(dnd::type_id_of<dst_type>::value));
        DND_ASSERT_ALIGNED(src, 0, sizeof(src_type),
                "src type: " << dnd::dtype(dnd::type_id_of<src_type>::value) << "  dst type id: " << dnd::dtype(dnd::type_id_of<dst_type>::value));
        //DEBUG_COUT << "multiple_assigner::assign_noexcept_contigstride_zerostride (" << typeid(src_type).name() << " -> " << typeid(dst_type).name() << ")\n";
        dst_type src_value_cached;
        dst_type *dst_cached = reinterpret_cast<dst_type *>(dst);

        single_assigner_builtin<dst_type, src_type, errmode>::assign(&src_value_cached, (const src_type *)src);

        for (intptr_t i = 0; i < count; ++i, ++dst_cached) {
            *dst_cached = src_value_cached;
        }
    }

    static void assign_contigstride_contigstride(char *dst, intptr_t,
                                const char *src, intptr_t,
                                intptr_t count,
                                const AuxDataBase *)
    {
        DND_ASSERT_ALIGNED(dst, sizeof(dst_type), sizeof(dst_type),
                "src type: " << dnd::dtype(dnd::type_id_of<src_type>::value) << "  dst type id: " << dnd::dtype(dnd::type_id_of<dst_type>::value));
        DND_ASSERT_ALIGNED(src, sizeof(src_type), sizeof(src_type),
                "src type: " << dnd::dtype(dnd::type_id_of<src_type>::value) << "  dst type id: " << dnd::dtype(dnd::type_id_of<dst_type>::value));
        //DEBUG_COUT << "multiple_assigner::assign_noexcept_contigstride_contigstride (" << typeid(src_type).name() << " -> " << typeid(dst_type).name() << ")\n";
        const src_type *src_cached = reinterpret_cast<const src_type *>(src);
        dst_type *dst_cached = reinterpret_cast<dst_type *>(dst);

        for (intptr_t i = 0; i < count; ++i, ++dst_cached, ++src_cached) {
            single_assigner_builtin<dst_type, src_type, errmode>::assign(dst_cached, src_cached);
        }
    }
};

} // namespace dnd

#endif // _DND__MULTIPLE_ASSIGNER_BUILTIN_HPP_
