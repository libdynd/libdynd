//
// Copyright (C) 2011-14 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/array.hpp>
#include <dynd/types/option_type.hpp>
#include <dynd/types/arrfunc_type.hpp>
#include <dynd/memblock/pod_memory_block.hpp>
#include <dynd/kernels/string_assignment_kernels.hpp>
#include <dynd/kernels/assignment_kernels.hpp>
#include <dynd/func/make_callable.hpp>
#include <dynd/pp/list.hpp>

#include <algorithm>

using namespace std;
using namespace dynd;

option_type::option_type(const ndt::type& value_tp)
    : base_type(option_type_id, option_kind, value_tp.get_data_size(),
                    value_tp.get_data_alignment(),
                    (value_tp.get_flags()&type_flags_value_inherited) | type_flag_constructor,
                    value_tp.get_metadata_size(),
                    value_tp.get_ndim()),
                    m_value_tp(value_tp)
{
    if (value_tp.is_builtin()) {
        
    }

    stringstream ss;
    ss << "cannot create type option[" << value_tp << "]";
    throw type_error(ss.str());
}

option_type::~option_type()
{
}

const ndt::type &option_type::make_nafunc_type()
{
    static ndt::type static_instance = ndt::make_cstruct(
        ndt::make_arrfunc(), "is_avail", ndt::make_arrfunc(), "assign_na");
    return static_instance;
}

void option_type::print_data(std::ostream &o, const char *arrmeta,
                             const char *data) const
{
    throw runtime_error("TODO: option_type::print_data");
}

void option_type::print_type(std::ostream& o) const
{
    o << "?" << m_value_tp;
}

bool option_type::is_expression() const
{
    // Even though the pointer is an instance of an base_expression_type,
    // we'll only call it an expression if the target is.
    return m_value_tp.is_expression();
}

bool option_type::is_unique_data_owner(const char *arrmeta) const
{
    if (m_value_tp.get_flags()&type_flag_blockref) {
        return m_value_tp.extended()->is_unique_data_owner(arrmeta);
    }
    return true;
}

void option_type::transform_child_types(type_transform_fn_t transform_fn,
                                        void *extra,
                                        ndt::type &out_transformed_tp,
                                        bool &out_was_transformed) const
{
    ndt::type tmp_tp;
    bool was_transformed = false;
    transform_fn(m_value_tp, extra, tmp_tp, was_transformed);
    if (was_transformed) {
        out_transformed_tp = ndt::make_option(tmp_tp);
        out_was_transformed = true;
    } else {
        out_transformed_tp = ndt::type(this, true);
    }
}


ndt::type option_type::get_canonical_type() const
{
    return ndt::make_option(m_value_tp.get_canonical_type());
}

bool option_type::is_lossless_assignment(const ndt::type& dst_tp, const ndt::type& src_tp) const
{
    if (dst_tp.extended() == this) {
        return ::is_lossless_assignment(m_value_tp, src_tp);
    } else {
        return ::is_lossless_assignment(dst_tp, m_value_tp);
    }
}

bool option_type::operator==(const base_type& rhs) const
{
    if (this == &rhs) {
        return true;
    } else if (rhs.get_type_id() != option_type_id) {
        return false;
    } else {
        const option_type *ot = static_cast<const option_type*>(&rhs);
        return m_value_tp == ot->m_value_tp;
    }
}

void option_type::metadata_default_construct(char *arrmeta, intptr_t ndim,
                                             const intptr_t *shape) const
{
    if (!m_value_tp.is_builtin()) {
        m_value_tp.extended()->metadata_default_construct(arrmeta, ndim, shape);
    }
}

void option_type::metadata_copy_construct(char *dst_arrmeta, const char *src_arrmeta, memory_block_data *embedded_reference) const
{
    if (!m_value_tp.is_builtin()) {
        m_value_tp.extended()->metadata_copy_construct(dst_arrmeta, src_arrmeta,
                                                       embedded_reference);
    }
}

void option_type::metadata_reset_buffers(char *arrmeta) const
{
    if (!m_value_tp.is_builtin()) {
        m_value_tp.extended()->metadata_reset_buffers(arrmeta);
    }
}

void option_type::metadata_finalize_buffers(char *arrmeta) const
{
    if (!m_value_tp.is_builtin()) {
        m_value_tp.extended()->metadata_finalize_buffers(arrmeta);
    }
}

void option_type::metadata_destruct(char *arrmeta) const
{
    if (!m_value_tp.is_builtin()) {
        m_value_tp.extended()->metadata_destruct(arrmeta);
    }
}

void option_type::metadata_debug_print(const char *arrmeta, std::ostream& o, const std::string& indent) const
{
    o << indent << "option arrmeta\n";
    if (!m_value_tp.is_builtin()) {
        m_value_tp.extended()->metadata_debug_print(arrmeta, o, indent + " ");
    }
}

static ndt::type property_get_value_type(const ndt::type& tp) {
    const option_type *pd = tp.tcast<option_type>();
    return pd->get_value_type();
}

void option_type::get_dynamic_type_properties(
                const std::pair<std::string, gfunc::callable> **out_properties,
                size_t *out_count) const
{
    static pair<string, gfunc::callable> type_properties[] = {
        pair<string, gfunc::callable>(
            "value_type",
            gfunc::make_callable(&property_get_value_type, "self"))};

    *out_properties = type_properties;
    *out_count = sizeof(type_properties) / sizeof(type_properties[0]);
}

namespace {
    // TODO: use the PP meta stuff, but DYND_PP_LEN_MAX is set to 8 right now, would need to be 19
    struct static_pointer {
        option_type bt1;
        option_type bt2;
        option_type bt3;
        option_type bt4;
        option_type bt5;
        option_type bt6;
        option_type bt7;
        option_type bt8;
        option_type bt9;
        option_type bt10;
        option_type bt11;
        option_type bt12;
        option_type bt13;
        option_type bt14;
        option_type bt15;
        option_type bt16;
        option_type bt17;
        option_type bt18;

        ndt::type static_builtins_instance[builtin_type_id_count];

        static_pointer()
            : bt1(ndt::type((type_id_t)1)),
              bt2(ndt::type((type_id_t)2)),
              bt3(ndt::type((type_id_t)3)),
              bt4(ndt::type((type_id_t)4)),
              bt5(ndt::type((type_id_t)5)),
              bt6(ndt::type((type_id_t)6)),
              bt7(ndt::type((type_id_t)7)),
              bt8(ndt::type((type_id_t)8)),
              bt9(ndt::type((type_id_t)9)),
              bt10(ndt::type((type_id_t)10)),
              bt11(ndt::type((type_id_t)11)),
              bt12(ndt::type((type_id_t)12)),
              bt13(ndt::type((type_id_t)13)),
              bt14(ndt::type((type_id_t)14)),
              bt15(ndt::type((type_id_t)15)),
              bt16(ndt::type((type_id_t)16)),
              bt17(ndt::type((type_id_t)17)),
              bt18(ndt::type((type_id_t)18))
        {
            static_builtins_instance[1] = ndt::type(&bt1, true);
            static_builtins_instance[2] = ndt::type(&bt2, true);
            static_builtins_instance[3] = ndt::type(&bt3, true);
            static_builtins_instance[4] = ndt::type(&bt4, true);
            static_builtins_instance[5] = ndt::type(&bt5, true);
            static_builtins_instance[6] = ndt::type(&bt6, true);
            static_builtins_instance[7] = ndt::type(&bt7, true);
            static_builtins_instance[8] = ndt::type(&bt8, true);
            static_builtins_instance[9] = ndt::type(&bt9, true);
            static_builtins_instance[10] = ndt::type(&bt10, true);
            static_builtins_instance[11] = ndt::type(&bt11, true);
            static_builtins_instance[12] = ndt::type(&bt12, true);
            static_builtins_instance[13] = ndt::type(&bt13, true);
            static_builtins_instance[14] = ndt::type(&bt14, true);
            static_builtins_instance[15] = ndt::type(&bt15, true);
            static_builtins_instance[16] = ndt::type(&bt16, true);
            static_builtins_instance[17] = ndt::type(&bt17, true);
            static_builtins_instance[18] = ndt::type(&bt18, true);
        }
    };
} // anonymous namespace

ndt::type ndt::make_option(const ndt::type& value_tp)
{
    // Static instances of the type, which have a reference
    // count > 0 for the lifetime of the program. This static
    // construction is inside a function to ensure correct creation
    // order during startup.
    static static_pointer sp;

    if (value_tp.is_builtin()) {
        return sp.static_builtins_instance[value_tp.get_type_id()];
    } else {
        return ndt::type(new option_type(value_tp), false);
    }
}
