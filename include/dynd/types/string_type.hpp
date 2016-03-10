//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//
// The string type uses memory_block references to store
// arbitrarily sized strings.
//

#pragma once

#include <dynd/bytes.hpp>
#include <dynd/type.hpp>
#include <dynd/string_encodings.hpp>

namespace dynd {

class DYNDT_API string : public bytes {
public:
  string() {}

  string(const char *data, size_t size) : bytes(data, size) {}

  string(const std::string &other) : string(other.data(), other.size()) {}

  bool operator<(const string &rhs) const
  {
    return std::lexicographical_compare(begin(), end(), rhs.begin(), rhs.end());
  }

  bool operator<=(const string &rhs) const
  {
    return !std::lexicographical_compare(rhs.begin(), rhs.end(), begin(), end());
  }

  bool operator>=(const string &rhs) const
  {
    return !std::lexicographical_compare(begin(), end(), rhs.begin(), rhs.end());
  }

  bool operator>(const string &rhs) const
  {
    return std::lexicographical_compare(rhs.begin(), rhs.end(), begin(), end());
  }

  const string operator+(const string &rhs)
  {
    string result;

    result.resize(size() + rhs.size());

    DYND_MEMCPY(result.begin(), begin(), size());
    DYND_MEMCPY(result.begin() + size(), rhs.begin(), rhs.size());

    return result;
  }
};

namespace ndt {

  class DYNDT_API string_type : public base_string_type {
  private:
    const string_encoding_t m_encoding = string_encoding_utf_8;
  public:
    typedef string data_type;

    string_type();

    string_encoding_t get_encoding() const { return m_encoding; }

    /** Alignment of the string data being pointed to. */
    size_t get_target_alignment() const { return string_encoding_char_size_table[string_encoding_utf_8]; }

    void get_string_range(const char **out_begin, const char **out_end, const char *arrmeta, const char *data) const;
    void set_from_utf8_string(const char *arrmeta, char *dst, const char *utf8_begin, const char *utf8_end,
                              const eval::eval_context *ectx) const;

    void print_data(std::ostream &o, const char *arrmeta, const char *data) const;

    void print_type(std::ostream &o) const;

    bool is_unique_data_owner(const char *arrmeta) const;
    type get_canonical_type() const;

    void get_shape(intptr_t ndim, intptr_t i, intptr_t *out_shape, const char *arrmeta, const char *data) const;

    std::map<std::string, std::pair<ndt::type, const char *>> get_dynamic_type_properties() const;

    bool is_lossless_assignment(const type &dst_tp, const type &src_tp) const;

    bool operator==(const base_type &rhs) const;

    void data_destruct(const char *arrmeta, char *data) const;
    void data_destruct_strided(const char *arrmeta, char *data, intptr_t stride, size_t count) const;
  };

  template <>
  struct traits<string> {
    static const size_t ndim = 0;

    static const bool is_same_layout = true;

    static type equivalent() { return type(string_id); }

    static string na() { return string(); }
  };

  template <>
  struct traits<std::string> {
    static const size_t ndim = 0;

    static const bool is_same_layout = false;

    static type equivalent() { return make_type<string>(); }
  };

} // namespace dynd::ndt
} // namespace dynd
