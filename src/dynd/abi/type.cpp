#include "dynd/abi/type.h"

extern "C" {

DYND_ABI_EXPORT dynd_type_range dynd_type_range_empty(dynd_type_header*) dynd_noexcept {
  return {nullptr, nullptr};
}

}
