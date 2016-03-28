#include <dynd/functional.hpp>

namespace dynd {
namespace nd {
  namespace functional {
    [[deprecated("Using reduction from the header <dynd/func/reduction.hpp> is deprecated. Please stop using that "
                 "header. Elwise "
                 "is now provided in <dynd/functional.hpp>.")]] DYND_API callable
    reduction(const callable &child);
  } // namespace dynd::nd::functional
} // namespace dynd::nd
} // namespace dynd
