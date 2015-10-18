//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/string.hpp>

using namespace std;
using namespace dynd;

bool dynd::operator<(const string &lhs, const string &rhs)
{
  return std::lexicographical_compare(
      reinterpret_cast<const uint8_t *>(lhs.begin()), reinterpret_cast<const uint8_t *>(lhs.end()),
      reinterpret_cast<const uint8_t *>(rhs.begin()), reinterpret_cast<const uint8_t *>(rhs.end()));
}

bool dynd::operator<=(const string &lhs, const string &rhs)
{
  return !std::lexicographical_compare(
              reinterpret_cast<const uint8_t *>(rhs.begin()), reinterpret_cast<const uint8_t *>(rhs.end()),
              reinterpret_cast<const uint8_t *>(lhs.begin()), reinterpret_cast<const uint8_t *>(lhs.end()));
}

bool dynd::operator==(const string &lhs, const string &rhs)
{
  return (lhs.end() - lhs.begin() == rhs.end() - rhs.begin()) &&
         memcmp(lhs.begin(), rhs.begin(), lhs.end() - lhs.begin()) == 0;
}

bool dynd::operator!=(const string &lhs, const string &rhs)
{
  return (lhs.end() - lhs.begin() != rhs.end() - rhs.begin()) ||
         memcmp(lhs.begin(), rhs.begin(), lhs.end() - lhs.begin()) != 0;
}

bool dynd::operator>=(const string &lhs, const string &rhs)
{
  return !std::lexicographical_compare(
              reinterpret_cast<const uint8_t *>(lhs.begin()), reinterpret_cast<const uint8_t *>(lhs.end()),
              reinterpret_cast<const uint8_t *>(rhs.begin()), reinterpret_cast<const uint8_t *>(rhs.end()));
}

bool dynd::operator>(const string &lhs, const string &rhs)
{
  return std::lexicographical_compare(
      reinterpret_cast<const uint8_t *>(rhs.begin()), reinterpret_cast<const uint8_t *>(rhs.end()),
      reinterpret_cast<const uint8_t *>(lhs.begin()), reinterpret_cast<const uint8_t *>(lhs.end()));
}
