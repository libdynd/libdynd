#include <dynd/func/apply.hpp>
#include <dynd/func/random.hpp>

using namespace std;
using namespace dynd;

nd::arrfunc nd::decl::uniform::make()
{
  return as_arrfunc<kernels::uniform_ck, std::default_random_engine,
                    integral_types>(
      ndt::type("(a: ?R, b: ?R, tp: type | R) -> R"),
      std::shared_ptr<std::default_random_engine>(
          new std::default_random_engine()));
}

nd::decl::uniform nd::uniform;