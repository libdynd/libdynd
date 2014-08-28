//
// Copyright (C) 2011-14 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DYND__JSON_TYPE_HPP_
#define _DYND__JSON_TYPE_HPP_

#include <dynd/types/string_type.hpp>

namespace dynd {

// The json type is stored as a string, but limited to
// UTF-8 and is supposed to contain JSON data.
typedef string_type_arrmeta json_type_arrmeta;
typedef string_type_data json_type_data;

class json_type : public base_string_type {
public:
    json_type();

    virtual ~json_type();

    string_encoding_t get_encoding() const {
        return string_encoding_utf_8;
    }

    void get_string_range(const char **out_begin, const char **out_end,
                          const char *arrmeta, const char *data) const;
    void set_from_utf8_string(const char *arrmeta, char *dst,
                              const char *utf8_begin, const char *utf8_end,
                              const eval::eval_context *ectx) const;

    void print_data(std::ostream& o, const char *arrmeta, const char *data) const;

    void print_type(std::ostream& o) const;

    bool is_unique_data_owner(const char *arrmeta) const;
    ndt::type get_canonical_type() const;

    bool is_lossless_assignment(const ndt::type& dst_tp, const ndt::type& src_tp) const;

    bool operator==(const base_type& rhs) const;

    void arrmeta_default_construct(char *arrmeta, intptr_t ndim,
                                   const intptr_t *shape,
                                   bool blockref_alloc) const;
    void arrmeta_copy_construct(char *dst_arrmeta, const char *src_arrmeta, memory_block_data *embedded_reference) const;
    void arrmeta_reset_buffers(char *arrmeta) const;
    void arrmeta_finalize_buffers(char *arrmeta) const;
    void arrmeta_destruct(char *arrmeta) const;
    void arrmeta_debug_print(const char *arrmeta, std::ostream& o, const std::string& indent) const;

    size_t make_assignment_kernel(ckernel_builder *ckb, intptr_t ckb_offset,
                                  const ndt::type &dst_tp,
                                  const char *dst_arrmeta,
                                  const ndt::type &src_tp,
                                  const char *src_arrmeta,
                                  kernel_request_t kernreq,
                                  const eval::eval_context *ectx) const;

    void make_string_iter(dim_iter *out_di, string_encoding_t encoding,
            const char *arrmeta, const char *data,
            const memory_block_ptr& ref,
            intptr_t buffer_max_mem,
            const eval::eval_context *ectx) const;
};

namespace ndt {
  inline ndt::type make_json()
  {
    return *reinterpret_cast<const ndt::type *>(&types::json_tp);
  }
} // namespace ndt

} // namespace dynd

#endif // _DYND__JSON_TYPE_HPP_

