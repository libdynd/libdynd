//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <bitset>

#include <dynd/type.hpp>

namespace dynd {

/**
 * Function prototype for a function which parses the type constructor arguments. It is called on the range of bytes
 * starting from the '[' character. If there is a '*' after the corresponding ']', this function is responsible
 * for handling it.
 */
typedef ndt::type (*low_level_type_args_parse_fn_t)(type_id_t id, const char *&begin, const char *end,
                                                    std::map<std::string, ndt::type> &symtable);
/**
 * Function prototype for a type constructor.
 *
 * \param id  The type id of of the type being constructed.
 * \param args  The type constructor arguments. e.g. as produced by `dynd::parse_type_constr_args`.
 * \param element_type  Provided if and only if the type is being constructed as a dimension type, is the
 *                      element type of the dimension.
 */
typedef ndt::type (*type_constructor_fn_t)(type_id_t id, const nd::buffer &args, const ndt::type &element_type);

/**
 * Default mechanism for parsing type arguments, when an optimized version is not implemented. It parses the type
 * arguments via `dynd::parse_type_constr_args`, then calls the type constructor for the specified type id.
 */
DYNDT_API ndt::type default_parse_type_args(type_id_t id, const char *&begin, const char *end,
                                            std::map<std::string, ndt::type> &symtable);

struct id_info {
  /** The name to use for parsing as a singleton or constructed type */
  std::string name;
  type_id_t base_id;
  /**
   * The type for parsing with no type constructor.
   *
   * If this is has uninitialized_type_id, i.e. is ndt::type(), then the type requires a type constructor
   */
  ndt::type singleton_type;
  /**
   * High-level generic construction from an nd::buffer of type arguments. Either both `parse_type_args` and
   * `construct_type` must be provided, or both must be NULL.
   *
   * If this is NULL, then the type cannot be created with a type constructor and requires a singleton type.
   */
  type_constructor_fn_t construct_type;
  /**
   * Low-level optimized type arguments parser. Must be equivalent to `dynd::parse_type_constr_args` followed by calling
   * the type constructor. May be set to `dynd::default_parse_type_args`.
   *
   * If this is NULL, then the type cannot be created with a type constructor and requires a singleton type.
   */
  low_level_type_args_parse_fn_t parse_type_args;

  id_info() = default;
  id_info(const id_info &) = default;
  id_info(id_info &&) = default;

  id_info(const char *name, type_id_t base_id, ndt::type &&singleton_type, type_constructor_fn_t construct_type,
          low_level_type_args_parse_fn_t parse_type_args)
      : name(name), base_id(base_id), singleton_type(singleton_type), construct_type(construct_type),
        parse_type_args(parse_type_args) {}
};

namespace detail {

  extern DYNDT_API std::vector<id_info> &infos();

} // namespace dynd::detail

DYNDT_API type_id_t new_id(const char *name, type_id_t base_id);

/**
 * Searches for a name in the type id registry, returning the `id_info` corresponding to it.
 *
 * \return The type id and its corresponding id_info, uninitialized_id when the name isn't found.
 */
DYNDT_API std::pair<type_id_t, const id_info *> lookup_id_by_name(const std::string &name);

/**
 * For type ids which are pre-allocated, but whose implementation isn't part of dynd.ndt, this function provides the
 * mechanism to add the type construction and parsing.
 */
DYNDT_API void register_known_type_id_constructor(type_id_t id, ndt::type &&singleton_type,
                                                  type_constructor_fn_t construct_type,
                                                  low_level_type_args_parse_fn_t parse_type_args = nullptr);

} // namespace dynd
