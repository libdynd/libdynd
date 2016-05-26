//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <string>
#include <vector>

#include <dynd/types/base_type.hpp>
#include <dynd/types/typevar_type.hpp>

namespace dynd {
namespace ndt {

  class DYNDT_API typevar_constructed_type : public base_type {
    std::string m_name;
    type m_arg;

  public:
    typevar_constructed_type(type_id_t new_id, const std::string &name, const type &arg)
        : base_type(new_id, typevar_constructed_id, 0, 1, type_flag_symbolic, 0, arg.get_ndim(),
                    arg.get_strided_ndim()),
          m_name(name), m_arg(arg) {
      //  static ndt::type args_pattern("((...), {...})");
      if (m_name.empty()) {
        throw type_error("dynd typevar name cannot be null");
      } else if (!is_valid_typevar_name(m_name.c_str(), m_name.c_str() + m_name.size())) {
        std::stringstream ss;
        ss << "dynd typevar name ";
        print_escaped_utf8_string(ss, m_name);
        ss << " is not valid, it must be alphanumeric and begin with a capital";
        throw type_error(ss.str());
      }
      // else if (!args.get_type().match(args_pattern)) {
      //  stringstream ss;
      // ss << "dynd constructed typevar must have args matching " << args_pattern
      // << ", which " << args.get_type() << " does not";
      // throw type_error(ss.str());
      //}
    }

    std::string get_name() const { return m_name; }

    type get_arg() const { return m_arg; }

    void get_vars(std::unordered_set<std::string> &vars) const;

    void print_data(std::ostream &o, const char *arrmeta, const char *data) const;

    void print_type(std::ostream &o) const;

    intptr_t get_dim_size(const char *arrmeta, const char *data) const;

    type apply_linear_index(intptr_t nindices, const irange *indices, size_t current_i, const type &root_tp,
                            bool leading_dimension) const;
    intptr_t apply_linear_index(intptr_t nindices, const irange *indices, const char *arrmeta, const type &result_tp,
                                char *out_arrmeta, const nd::memory_block &embedded_reference, size_t current_i,
                                const type &root_tp, bool leading_dimension, char **inout_data,
                                nd::memory_block &inout_dataref) const;

    bool is_lossless_assignment(const type &dst_tp, const type &src_tp) const;

    bool operator==(const base_type &rhs) const;

    void arrmeta_default_construct(char *arrmeta, bool blockref_alloc) const;
    void arrmeta_copy_construct(char *dst_arrmeta, const char *src_arrmeta,
                                const nd::memory_block &embedded_reference) const;
    void arrmeta_destruct(char *arrmeta) const;

    bool match(const type &candidate_tp, std::map<std::string, type> &tp_vars) const;

    std::map<std::string, std::pair<ndt::type, const char *>> get_dynamic_type_properties() const;
  };

} // namespace dynd::ndt
} // namespace dynd
